#!/usr/bin/env python3
"""
Debug script to understand how the real router handles test queries.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append('/Users/Mark/Research/Clinical_Metabolomics_Oracle/smo_chatbot_August_6th_2025')
from lightrag_integration.query_router import BiomedicalQueryRouter, RoutingDecision
from lightrag_integration.tests.test_comprehensive_routing_decision_logic import TestDataGenerator

def debug_routing():
    """Debug routing decisions for test queries"""
    
    print("=== Debug Routing Test ===")
    
    # Create real router
    router = BiomedicalQueryRouter()
    test_generator = TestDataGenerator()
    
    # Test a few LIGHTRAG queries
    lightrag_queries = test_generator.generate_lightrag_queries(5)
    
    print("\n=== LIGHTRAG Test Queries ===")
    for i, test_case in enumerate(lightrag_queries):
        print(f"\nQuery {i+1}: {test_case.query}")
        print(f"Expected: {test_case.expected_route.value}")
        print(f"Expected confidence range: {test_case.expected_confidence_range}")
        
        result = router.route_query(test_case.query)
        print(f"Actual route: {result.routing_decision.value}")
        print(f"Actual confidence: {result.confidence:.3f}")
        print(f"Research category: {result.research_category.value}")
        print(f"Reasoning: {result.reasoning}")
        
        # Check if it matches
        if result.routing_decision == test_case.expected_route:
            print("✅ Routing decision matches")
        else:
            print("❌ Routing decision does not match")
        
        min_conf, max_conf = test_case.expected_confidence_range
        if min_conf <= result.confidence <= max_conf:
            print("✅ Confidence in range")
        else:
            print("❌ Confidence outside range")
    
    # Test PERPLEXITY queries
    perplexity_queries = test_generator.generate_perplexity_queries(3)
    
    print("\n=== PERPLEXITY Test Queries ===")
    for i, test_case in enumerate(perplexity_queries):
        print(f"\nQuery {i+1}: {test_case.query}")
        print(f"Expected: {test_case.expected_route.value}")
        
        result = router.route_query(test_case.query)
        print(f"Actual route: {result.routing_decision.value}")
        print(f"Actual confidence: {result.confidence:.3f}")
        print(f"Research category: {result.research_category.value}")
        
        if result.routing_decision == test_case.expected_route:
            print("✅ Routing decision matches")
        else:
            print("❌ Routing decision does not match")
    
    # Test some specific problematic queries
    print("\n=== Specific Problem Queries ===")
    problem_queries = [
        "How do obesity and insulin interact in regulation?",
        "What is the relationship between glucose and insulin?",
        "Latest metabolomics research 2025",
        "What are metabolomic pathways?",
        ""
    ]
    
    for query in problem_queries:
        print(f"\nQuery: '{query}'")
        result = router.route_query(query)
        print(f"Route: {result.routing_decision.value}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Category: {result.research_category.value}")
        print(f"Reasoning: {result.reasoning}")

if __name__ == "__main__":
    debug_routing()