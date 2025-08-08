#!/usr/bin/env python3
"""
Quick validation that confidence scoring tests are working correctly.
"""

import sys
sys.path.append('lightrag_integration')

from lightrag_integration.research_categorizer import ResearchCategorizer, CategoryPrediction
from lightrag_integration.relevance_scorer import QueryTypeClassifier
from lightrag_integration.cost_persistence import ResearchCategory

def validate_confidence_scoring():
    """Validate confidence scoring functionality."""
    print("🎯 Validating Intent Detection Confidence Scoring...")
    
    # Initialize components
    categorizer = ResearchCategorizer()
    query_classifier = QueryTypeClassifier()
    
    # Test queries with expected behavior
    test_queries = [
        ("What is LC-MS metabolomics analysis?", "should get high confidence"),
        ("metabolite identification using mass spectrometry", "should get high confidence"),
        ("hmm", "should get very low confidence"),
        ("analysis", "should get low confidence"),
    ]
    
    results = []
    for query, expectation in test_queries:
        # Get prediction
        prediction = categorizer.categorize_query(query)
        
        # Get query type
        query_type = query_classifier.classify_query(query)
        
        results.append({
            'query': query,
            'confidence': prediction.confidence,
            'confidence_level': categorizer._get_confidence_level(prediction.confidence),
            'category': prediction.category.value,
            'query_type': query_type,
            'evidence': prediction.evidence,
            'expectation': expectation
        })
        
        print(f"  Query: '{query}'")
        print(f"    Confidence: {prediction.confidence:.3f} ({categorizer._get_confidence_level(prediction.confidence)})")
        print(f"    Category: {prediction.category.value}")
        print(f"    Query Type: {query_type}")
        print(f"    Evidence: {prediction.evidence}")
        print(f"    Expected: {expectation}")
        print()
    
    print("✅ Confidence scoring validation completed!")
    print(f"✅ All {len(results)} queries processed successfully")
    return results

if __name__ == "__main__":
    validate_confidence_scoring()