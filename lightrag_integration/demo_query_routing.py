"""
Demonstration script for the Biomedical Query Router.

This script shows how the query routing system classifies different types
of biomedical queries and routes them appropriately between LightRAG and 
Perplexity API based on content analysis and temporal requirements.

Usage:
    python lightrag_integration/demo_query_routing.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag_integration.query_router import BiomedicalQueryRouter, RoutingDecision
import logging


def demo_routing_system():
    """Demonstrate the biomedical query routing system."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize router
    print("Initializing Biomedical Query Router...")
    router = BiomedicalQueryRouter(logger=logger)
    
    # Test queries from different categories
    test_queries = [
        # Knowledge graph queries (relationships, pathways, mechanisms)
        "What are the metabolic pathways involved in diabetes?",
        "Show me the relationship between metabolites and cardiovascular disease",
        "How do biomarkers connect to clinical diagnosis?",
        "Mechanism of metabolite identification using mass spectrometry",
        "What pathways are affected in cancer metabolism?",
        
        # Real-time queries (latest, recent, current)
        "What are the latest metabolomics research findings?",
        "Recent advances in clinical metabolomics published in 2024",
        "Current trends in biomarker discovery",
        "Breaking news in drug discovery",
        "What's new in personalized medicine this year?",
        
        # General queries (what is, define, explain)
        "What is clinical metabolomics?",
        "Define biomarker discovery",
        "Explain pathway analysis methodology",
        "Give me an overview of mass spectrometry",
        
        # Mixed/complex queries
        "What are the latest advances in metabolic pathway analysis published this year?",
        "How do recent biomarker discoveries relate to established disease mechanisms?",
        "Compare traditional metabolite identification methods with current approaches",
    ]
    
    print(f"\nTesting {len(test_queries)} queries...")
    print("=" * 80)
    
    # Process each query
    routing_stats = {"lightrag": 0, "perplexity": 0, "either": 0, "hybrid": 0}
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i:2d}. Query: {query}")
        print("-" * 60)
        
        # Route the query
        prediction = router.route_query(query)
        
        # Display results
        print(f"    Routing Decision: {prediction.routing_decision.value.upper()}")
        print(f"    Confidence:       {prediction.confidence:.3f}")
        print(f"    Research Category: {prediction.research_category.value}")
        
        # Show reasoning
        print(f"    Reasoning:")
        for reason in prediction.reasoning:
            print(f"      • {reason}")
        
        # Show temporal indicators if any
        if prediction.temporal_indicators:
            print(f"    Temporal Indicators: {', '.join(prediction.temporal_indicators)}")
        
        # Update statistics
        routing_stats[prediction.routing_decision.value] += 1
    
    # Display summary statistics
    print("\n" + "=" * 80)
    print("ROUTING SUMMARY")
    print("=" * 80)
    
    total_queries = len(test_queries)
    for routing, count in routing_stats.items():
        percentage = (count / total_queries) * 100
        print(f"{routing.upper():10s}: {count:2d} queries ({percentage:5.1f}%)")
    
    print(f"\nTotal processed: {total_queries} queries")
    
    # Show routing mappings
    print("\n" + "=" * 80)
    print("CATEGORY ROUTING MAPPINGS")
    print("=" * 80)
    
    category_mappings = router.category_routing_map
    for category, routing in sorted(category_mappings.items(), key=lambda x: x[0].value):
        print(f"{category.value:25s} -> {routing.value.upper()}")
    
    # Demonstrate helper methods
    print("\n" + "=" * 80)
    print("HELPER METHODS DEMONSTRATION")
    print("=" * 80)
    
    helper_test_queries = [
        "What are metabolic pathways in diabetes?",
        "Latest research in metabolomics 2024",
        "Define biomarker discovery"
    ]
    
    for query in helper_test_queries:
        should_lightrag = router.should_use_lightrag(query)
        should_perplexity = router.should_use_perplexity(query)
        
        print(f"\nQuery: {query}")
        print(f"  should_use_lightrag():   {should_lightrag}")
        print(f"  should_use_perplexity(): {should_perplexity}")
    
    # Show system statistics
    print("\n" + "=" * 80)
    print("SYSTEM STATISTICS")
    print("=" * 80)
    
    stats = router.get_routing_statistics()
    print(f"Total predictions made:     {stats['total_predictions']}")
    print(f"Average confidence:         {stats['average_confidence']:.3f}")
    print(f"Temporal keywords count:    {stats['temporal_keywords_count']}")
    print(f"Temporal patterns count:    {stats['temporal_patterns_count']}")
    
    print("\nRouting thresholds:")
    for threshold, value in stats['routing_thresholds'].items():
        print(f"  {threshold:20s}: {value:.3f}")


if __name__ == "__main__":
    demo_routing_system()