#!/usr/bin/env python3
"""
Debug script to test routing scores and identify the core issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lightrag_integration.query_router import BiomedicalQueryRouter, RoutingDecision
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def debug_single_query(query_text):
    """Debug routing for a single query"""
    router = BiomedicalQueryRouter()
    
    print(f"\n=== DEBUG: {query_text} ===")
    
    # Get full prediction with details
    prediction = router.route_query(query_text)
    
    print(f"Final Decision: {prediction.routing_decision.value}")
    print(f"Confidence: {prediction.confidence:.3f}")
    print(f"Research Category: {prediction.research_category.value}")
    
    print("\nDetailed Analysis:")
    if prediction.metadata and 'analysis_results' in prediction.metadata:
        analysis = prediction.metadata['analysis_results']
        
        # Temporal analysis
        if 'temporal_analysis' in analysis:
            temporal = analysis['temporal_analysis']
            print(f"  Temporal Score: {temporal.get('temporal_score', 0.0)}")
            print(f"  Temporal Keywords: {temporal.get('temporal_keywords_found', [])}")
            print(f"  Has Temporal Patterns: {temporal.get('has_temporal_patterns', False)}")
        
        # KG detection
        if 'kg_detection' in analysis:
            kg = analysis['kg_detection']
            print(f"  KG Confidence: {kg.get('confidence', 0.0)}")
            print(f"  Has KG Intent: {kg.get('has_kg_intent', False)}")
            print(f"  Biomedical Entities: {kg.get('biomedical_entities', [])}")
    
    print(f"\nReasoning: {prediction.reasoning}")
    
    return prediction

if __name__ == "__main__":
    # Test specific failing queries
    test_queries = [
        "Recent biomarker discoveries for cancer detection",
        "What are the latest biomarker discoveries and how do they relate to metabolic pathways?",
        "Current LC-MS approaches for understanding insulin signaling mechanisms"
    ]
    
    for query in test_queries:
        debug_single_query(query)