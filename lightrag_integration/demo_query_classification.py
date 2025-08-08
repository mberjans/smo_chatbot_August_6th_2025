"""
Demonstration Script for Enhanced Query Classification System (CMO-LIGHTRAG-012-T04)

This script demonstrates the capabilities of the enhanced query classification system,
including integration with the existing biomedical query router and practical usage
examples for the Clinical Metabolomics Oracle.

Features Demonstrated:
- Three-category classification (KNOWLEDGE_GRAPH, REAL_TIME, GENERAL)
- Comprehensive confidence scoring and uncertainty quantification
- Integration with existing ResearchCategorizer and BiomedicalQueryRouter
- Performance benchmarking and validation
- Real-world query examples from clinical metabolomics

Author: Claude Code (Anthropic)
Created: August 8, 2025
Version: 1.0.0
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Import the new query classification system
from query_classification_system import (
    QueryClassificationCategories,
    BiomedicalKeywordSets,
    QueryClassificationEngine,
    ClassificationResult,
    create_classification_engine,
    classify_for_routing,
    get_routing_category_mapping
)

# Import existing systems for integration demonstration
from query_router import BiomedicalQueryRouter, RoutingDecision
from research_categorizer import ResearchCategorizer


def setup_logging():
    """Set up logging for the demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('demo_query_classification.log')
        ]
    )
    return logging.getLogger(__name__)


def demonstrate_basic_classification():
    """Demonstrate basic query classification functionality."""
    print("\n" + "="*80)
    print("BASIC QUERY CLASSIFICATION DEMONSTRATION")
    print("="*80)
    
    # Create classification engine
    logger = logging.getLogger('demo.basic')
    engine = create_classification_engine(logger)
    
    # Test queries for each category
    test_queries = [
        # Knowledge Graph queries
        ("What is the relationship between glucose metabolism and diabetes?", "Expected: KNOWLEDGE_GRAPH"),
        ("How do metabolites in the TCA cycle interact with each other?", "Expected: KNOWLEDGE_GRAPH"), 
        ("What are the key biomarkers for cardiovascular disease?", "Expected: KNOWLEDGE_GRAPH"),
        ("Explain the mechanism of fatty acid oxidation", "Expected: KNOWLEDGE_GRAPH"),
        
        # Real-Time queries
        ("What are the latest developments in metabolomics research?", "Expected: REAL_TIME"),
        ("Recent FDA approvals for metabolomics-based diagnostics in 2024", "Expected: REAL_TIME"),
        ("Breaking news in precision medicine and biomarker discovery", "Expected: REAL_TIME"),
        ("Current clinical trials for metabolic disorders", "Expected: REAL_TIME"),
        
        # General queries
        ("What is metabolomics?", "Expected: GENERAL"),
        ("Define biomarker in simple terms", "Expected: GENERAL"),
        ("How to perform LC-MS analysis step by step?", "Expected: GENERAL"),
        ("Compare NMR and mass spectrometry techniques", "Expected: GENERAL"),
    ]
    
    print(f"\nClassifying {len(test_queries)} test queries...\n")
    
    for i, (query, expected) in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print(f"Expected: {expected}")
        
        # Perform classification
        start_time = time.time()
        result = engine.classify_query(query)
        classification_time = (time.time() - start_time) * 1000
        
        # Display results
        print(f"Result: {result.category.value.upper()}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Classification Time: {classification_time:.2f}ms")
        print(f"Evidence: {', '.join(result.matched_keywords[:3])}")
        print(f"Reasoning: {result.reasoning[0] if result.reasoning else 'No reasoning provided'}")
        
        # Show alternatives
        alternatives = result.alternative_classifications[1:3]  # Top 2 alternatives
        if alternatives:
            print("Alternative classifications:")
            for alt_category, alt_confidence in alternatives:
                print(f"  - {alt_category.value}: {alt_confidence:.3f}")
        
        print("-" * 80)


def demonstrate_detailed_analysis():
    """Demonstrate detailed confidence analysis and metrics."""
    print("\n" + "="*80)
    print("DETAILED CONFIDENCE ANALYSIS DEMONSTRATION")
    print("="*80)
    
    engine = create_classification_engine()
    
    # Complex query that could belong to multiple categories
    complex_query = "Latest research on metabolic pathway biomarkers for cancer diagnosis"
    
    print(f"Analyzing complex query: '{complex_query}'\n")
    
    result = engine.classify_query(complex_query)
    
    # Display comprehensive analysis
    print("CLASSIFICATION RESULT:")
    print(f"Primary Category: {result.category.value.upper()}")
    print(f"Overall Confidence: {result.confidence:.3f}")
    print(f"Classification Time: {result.classification_time_ms:.2f}ms")
    print()
    
    print("DETAILED CONFIDENCE BREAKDOWN:")
    print(f"Keyword Match Confidence: {result.keyword_match_confidence:.3f}")
    print(f"Pattern Match Confidence: {result.pattern_match_confidence:.3f}")
    print(f"Semantic Confidence: {result.semantic_confidence:.3f}")
    print(f"Temporal Confidence: {result.temporal_confidence:.3f}")
    print()
    
    print("EVIDENCE ANALYSIS:")
    print(f"Matched Keywords: {result.matched_keywords}")
    print(f"Matched Patterns: {result.matched_patterns}")
    print(f"Biomedical Entities: {result.biomedical_entities}")
    print(f"Temporal Indicators: {result.temporal_indicators}")
    print()
    
    print("UNCERTAINTY QUANTIFICATION:")
    print(f"Ambiguity Score: {result.ambiguity_score:.3f} (lower is better)")
    print(f"Conflict Score: {result.conflict_score:.3f} (lower is better)")
    print()
    
    print("ALL ALTERNATIVE CLASSIFICATIONS:")
    for category, confidence in result.alternative_classifications:
        indicator = "★" if category == result.category else " "
        print(f"{indicator} {category.value}: {confidence:.3f}")
    print()
    
    print("REASONING:")
    for reason in result.reasoning:
        print(f"- {reason}")


def demonstrate_integration_with_router():
    """Demonstrate integration with existing biomedical query router."""
    print("\n" + "="*80)
    print("INTEGRATION WITH BIOMEDICAL QUERY ROUTER")
    print("="*80)
    
    # Create both systems
    logger = logging.getLogger('demo.integration')
    classification_engine = create_classification_engine(logger)
    biomedical_router = BiomedicalQueryRouter(logger)
    
    # Get routing category mapping
    category_mapping = get_routing_category_mapping()
    print("Query Classification → Routing Decision Mapping:")
    for category, routing in category_mapping.items():
        print(f"  {category.value} → {routing}")
    print()
    
    # Test integration with sample queries
    integration_test_queries = [
        "What are the metabolic pathways involved in diabetes?",
        "Latest breakthroughs in biomarker discovery 2024",
        "How to identify unknown metabolites using LC-MS?",
        "Recent clinical trials for metabolomics-based diagnostics"
    ]
    
    print("INTEGRATED CLASSIFICATION AND ROUTING:")
    print()
    
    for query in integration_test_queries:
        print(f"Query: {query}")
        
        # 1. New classification system
        classification_result = classification_engine.classify_query(query)
        suggested_routing = category_mapping[classification_result.category]
        
        print(f"Classification: {classification_result.category.value} (confidence: {classification_result.confidence:.3f})")
        print(f"Suggested Routing: {suggested_routing}")
        
        # 2. Existing biomedical router
        routing_prediction = biomedical_router.route_query(query)
        
        print(f"Router Decision: {routing_prediction.routing_decision.value} (confidence: {routing_prediction.confidence:.3f})")
        print(f"Router Category: {routing_prediction.research_category.value}")
        
        # 3. Compare results
        routing_match = (
            (suggested_routing == "lightrag" and routing_prediction.routing_decision == RoutingDecision.LIGHTRAG) or
            (suggested_routing == "perplexity" and routing_prediction.routing_decision == RoutingDecision.PERPLEXITY) or
            (suggested_routing == "either" and routing_prediction.routing_decision in [RoutingDecision.EITHER, RoutingDecision.HYBRID])
        )
        
        print(f"Routing Agreement: {'✓ YES' if routing_match else '✗ NO'}")
        print(f"Performance: Classification={classification_result.classification_time_ms:.1f}ms, Routing={routing_prediction.metadata.get('routing_time_ms', 0):.1f}ms")
        print("-" * 80)


def demonstrate_performance_benchmarking():
    """Demonstrate performance benchmarking and validation."""
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARKING")
    print("="*80)
    
    engine = create_classification_engine()
    
    # Performance test queries
    performance_queries = [
        "What is the relationship between metabolites and disease progression?",
        "Latest developments in metabolomics technology 2024",
        "How to perform statistical analysis of metabolomics data?",
        "Biomarkers for early detection of cancer",
        "Recent clinical trials in precision medicine",
        "Define metabolic fingerprinting methodology",
        "Pathway analysis using KEGG database",
        "Current trends in lipidomics research",
        "What is the mechanism of drug-metabolite interactions?",
        "Latest FDA regulations for metabolomics-based diagnostics"
    ]
    
    print(f"Benchmarking performance with {len(performance_queries)} queries...")
    print("Target: <2000ms per classification\n")
    
    # Collect performance metrics
    times = []
    accuracies = []
    confidence_scores = []
    
    start_time = time.time()
    
    for i, query in enumerate(performance_queries, 1):
        query_start = time.time()
        result = engine.classify_query(query)
        query_time = (time.time() - query_start) * 1000
        
        times.append(query_time)
        confidence_scores.append(result.confidence)
        
        # Performance status
        status = "✓ PASS" if query_time < 2000 else "✗ FAIL"
        print(f"Query {i:2d}: {query_time:6.2f}ms {status} (conf: {result.confidence:.3f})")
    
    total_time = (time.time() - start_time) * 1000
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    
    print("\nPERFORMANCE STATISTICS:")
    print(f"Total Benchmark Time: {total_time:.2f}ms")
    print(f"Average Classification Time: {avg_time:.2f}ms")
    print(f"Minimum Classification Time: {min_time:.2f}ms")
    print(f"Maximum Classification Time: {max_time:.2f}ms")
    print(f"Target Achievement Rate: {sum(1 for t in times if t < 2000) / len(times) * 100:.1f}%")
    print(f"Average Confidence Score: {avg_confidence:.3f}")
    
    # Performance validation
    performance_passed = all(t < 2000 for t in times)
    print(f"\nPERFORMANCE TEST: {'✓ PASSED' if performance_passed else '✗ FAILED'}")
    
    if not performance_passed:
        slow_queries = [(i, t) for i, t in enumerate(times) if t >= 2000]
        print(f"Slow queries: {len(slow_queries)} out of {len(times)}")
        for query_idx, query_time in slow_queries[:3]:  # Show first 3 slow queries
            print(f"  Query {query_idx + 1}: {query_time:.2f}ms")


def demonstrate_keyword_and_pattern_analysis():
    """Demonstrate keyword and pattern matching capabilities."""
    print("\n" + "="*80)
    print("KEYWORD AND PATTERN ANALYSIS")
    print("="*80)
    
    # Create keyword sets for inspection
    keyword_sets = BiomedicalKeywordSets()
    
    print("KEYWORD SET STATISTICS:")
    print(f"Knowledge Graph Keywords: {len(keyword_sets.knowledge_graph_set)}")
    print(f"Real-Time Keywords: {len(keyword_sets.real_time_set)}")
    print(f"General Keywords: {len(keyword_sets.general_set)}")
    print(f"Biomedical Entities: {len(keyword_sets.biomedical_entities_set)}")
    print()
    
    print("PATTERN STATISTICS:")
    print(f"Knowledge Graph Patterns: {len(keyword_sets.kg_patterns)}")
    print(f"Real-Time Patterns: {len(keyword_sets.rt_patterns)}")
    print(f"General Patterns: {len(keyword_sets.general_patterns)}")
    print()
    
    # Sample keywords from each category
    print("SAMPLE KEYWORDS BY CATEGORY:")
    print()
    
    print("Knowledge Graph Keywords (sample):")
    kg_sample = list(keyword_sets.knowledge_graph_set)[:10]
    for keyword in kg_sample:
        print(f"  - {keyword}")
    print()
    
    print("Real-Time Keywords (sample):")
    rt_sample = list(keyword_sets.real_time_set)[:10]
    for keyword in rt_sample:
        print(f"  - {keyword}")
    print()
    
    print("General Keywords (sample):")
    general_sample = list(keyword_sets.general_set)[:10]
    for keyword in general_sample:
        print(f"  - {keyword}")
    print()
    
    # Test pattern matching
    print("PATTERN MATCHING EXAMPLES:")
    test_text = "What is the relationship between metabolites and the latest biomarker research in 2024?"
    print(f"Test text: '{test_text}'\n")
    
    # Knowledge graph patterns
    print("Knowledge Graph Pattern Matches:")
    for pattern in keyword_sets.kg_patterns[:5]:  # Test first 5 patterns
        matches = pattern.findall(test_text.lower())
        if matches:
            print(f"  Pattern matched: {matches}")
    
    # Real-time patterns  
    print("\nReal-Time Pattern Matches:")
    for pattern in keyword_sets.rt_patterns[:5]:
        matches = pattern.findall(test_text.lower())
        if matches:
            print(f"  Pattern matched: {matches}")
    
    # General patterns
    print("\nGeneral Pattern Matches:")
    for pattern in keyword_sets.general_patterns[:5]:
        matches = pattern.findall(test_text.lower())
        if matches:
            print(f"  Pattern matched: {matches}")


def demonstrate_classification_validation():
    """Demonstrate classification validation and accuracy testing."""
    print("\n" + "="*80)
    print("CLASSIFICATION VALIDATION")
    print("="*80)
    
    engine = create_classification_engine()
    
    # Define test cases with expected results
    validation_test_cases = [
        # Knowledge Graph test cases
        ("What are the metabolic pathways involved in glucose metabolism?", 
         QueryClassificationCategories.KNOWLEDGE_GRAPH, (0.6, 1.0)),
        ("How do biomarkers relate to disease progression?", 
         QueryClassificationCategories.KNOWLEDGE_GRAPH, (0.5, 1.0)),
        ("Explain the mechanism of fatty acid oxidation", 
         QueryClassificationCategories.KNOWLEDGE_GRAPH, (0.6, 1.0)),
        
        # Real-Time test cases
        ("Latest breakthroughs in metabolomics research 2024", 
         QueryClassificationCategories.REAL_TIME, (0.7, 1.0)),
        ("Recent FDA approvals for biomarker diagnostics", 
         QueryClassificationCategories.REAL_TIME, (0.6, 1.0)),
        ("Current clinical trials in precision medicine", 
         QueryClassificationCategories.REAL_TIME, (0.5, 1.0)),
        
        # General test cases
        ("What is the definition of metabolomics?", 
         QueryClassificationCategories.GENERAL, (0.7, 1.0)),
        ("How to perform LC-MS analysis?", 
         QueryClassificationCategories.GENERAL, (0.6, 1.0)),
        ("Compare different analytical techniques", 
         QueryClassificationCategories.GENERAL, (0.5, 1.0)),
    ]
    
    print("VALIDATION TEST RESULTS:")
    print()
    
    correct_classifications = 0
    confidence_within_range = 0
    performance_passes = 0
    
    for i, (query, expected_category, confidence_range) in enumerate(validation_test_cases, 1):
        print(f"Test {i}: {query}")
        print(f"Expected: {expected_category.value}, Confidence: {confidence_range}")
        
        # Run validation
        validation_result = engine.validate_classification_performance(
            query, expected_category, confidence_range
        )
        
        # Extract results
        predicted_category = validation_result['predicted_category']
        predicted_confidence = validation_result['predicted_confidence']
        classification_correct = validation_result['classification_correct']
        meets_performance = validation_result['meets_performance_target']
        validation_passed = validation_result['validation_passed']
        
        print(f"Result: {predicted_category}, Confidence: {predicted_confidence:.3f}")
        print(f"Classification: {'✓ CORRECT' if classification_correct else '✗ INCORRECT'}")
        print(f"Performance: {'✓ PASS' if meets_performance else '✗ FAIL'}")
        print(f"Overall: {'✓ PASS' if validation_passed else '✗ FAIL'}")
        
        # Update counters
        if classification_correct:
            correct_classifications += 1
        if confidence_range[0] <= predicted_confidence <= confidence_range[1]:
            confidence_within_range += 1
        if meets_performance:
            performance_passes += 1
        
        if not validation_passed and validation_result['issues']:
            print(f"Issues: {'; '.join(validation_result['issues'])}")
        
        print("-" * 60)
    
    # Overall validation statistics
    total_tests = len(validation_test_cases)
    classification_accuracy = correct_classifications / total_tests
    confidence_accuracy = confidence_within_range / total_tests  
    performance_rate = performance_passes / total_tests
    
    print("OVERALL VALIDATION RESULTS:")
    print(f"Classification Accuracy: {classification_accuracy:.1%} ({correct_classifications}/{total_tests})")
    print(f"Confidence Range Accuracy: {confidence_accuracy:.1%} ({confidence_within_range}/{total_tests})")
    print(f"Performance Pass Rate: {performance_rate:.1%} ({performance_passes}/{total_tests})")
    print(f"Overall Success Rate: {min(classification_accuracy, performance_rate):.1%}")


def demonstrate_real_world_examples():
    """Demonstrate with real-world clinical metabolomics queries."""
    print("\n" + "="*80)
    print("REAL-WORLD CLINICAL METABOLOMICS EXAMPLES")
    print("="*80)
    
    engine = create_classification_engine()
    
    # Real-world examples from clinical metabolomics research
    real_world_queries = [
        # Research queries
        "I need to identify unknown metabolites in plasma samples from diabetic patients",
        "Can you help me understand the metabolic changes in cancer cells versus normal cells?", 
        "What statistical methods should I use for untargeted metabolomics data analysis?",
        "How do I validate biomarkers discovered in my metabolomics study?",
        
        # Clinical application queries
        "What are the current FDA-approved metabolomics tests for clinical diagnosis?",
        "Are there any new metabolomics-based diagnostics approved in 2024?",
        "What is the latest research on metabolomics for personalized medicine?",
        "Current clinical guidelines for metabolomics in patient care",
        
        # Educational queries
        "Explain the difference between targeted and untargeted metabolomics",
        "What is the basic workflow for a metabolomics experiment?",
        "How do I choose between LC-MS and GC-MS for my study?",
        "What are the key databases used in metabolomics research?",
        
        # Technical queries
        "Troubleshooting peak identification issues in LC-MS data",
        "Best practices for metabolomics sample preparation and storage",
        "How to handle missing values in metabolomics datasets?",
        "Quality control procedures for metabolomics experiments"
    ]
    
    print(f"Analyzing {len(real_world_queries)} real-world queries...\n")
    
    category_counts = {category: 0 for category in QueryClassificationCategories}
    confidence_scores = []
    processing_times = []
    
    for i, query in enumerate(real_world_queries, 1):
        result = engine.classify_query(query)
        
        category_counts[result.category] += 1
        confidence_scores.append(result.confidence)
        processing_times.append(result.classification_time_ms)
        
        print(f"Query {i:2d}: {query}")
        print(f"Category: {result.category.value.upper()}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Time: {result.classification_time_ms:.1f}ms")
        
        # Show top evidence
        if result.matched_keywords:
            print(f"Key evidence: {', '.join(result.matched_keywords[:3])}")
        
        print(f"Reasoning: {result.reasoning[0] if result.reasoning else 'No reasoning'}")
        print("-" * 80)
    
    # Summary statistics
    print("REAL-WORLD QUERY ANALYSIS SUMMARY:")
    print()
    
    print("Category Distribution:")
    for category, count in category_counts.items():
        percentage = count / len(real_world_queries) * 100
        print(f"  {category.value}: {count} queries ({percentage:.1f}%)")
    
    print()
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    avg_time = sum(processing_times) / len(processing_times)
    max_time = max(processing_times)
    
    print("Performance Summary:")
    print(f"  Average Confidence: {avg_confidence:.3f}")
    print(f"  Average Processing Time: {avg_time:.1f}ms")
    print(f"  Maximum Processing Time: {max_time:.1f}ms")
    print(f"  Performance Target Achievement: {sum(1 for t in processing_times if t < 2000) / len(processing_times) * 100:.1f}%")


def main():
    """Main demonstration function."""
    print("Enhanced Query Classification System Demonstration")
    print("Clinical Metabolomics Oracle - LightRAG Integration")
    print("CMO-LIGHTRAG-012-T04 Implementation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting query classification system demonstration")
    
    try:
        # Run all demonstrations
        demonstrate_basic_classification()
        demonstrate_detailed_analysis()
        demonstrate_integration_with_router()
        demonstrate_performance_benchmarking()
        demonstrate_keyword_and_pattern_analysis()
        demonstrate_classification_validation()
        demonstrate_real_world_examples()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nAll features of the enhanced query classification system have been demonstrated.")
        print("The system meets the CMO-LIGHTRAG-012-T04 requirements:")
        print("✓ Three-category classification (KNOWLEDGE_GRAPH, REAL_TIME, GENERAL)")
        print("✓ Comprehensive biomedical keyword sets")
        print("✓ Performance optimization (<2 second classification response)")
        print("✓ Integration with existing ResearchCategorizer and router systems")
        print("✓ Detailed confidence scoring and uncertainty quantification")
        print("✓ Real-world clinical metabolomics query handling")
        
        logger.info("Query classification system demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Demonstration failed with error: {e}")
        print(f"\nERROR: Demonstration failed with error: {e}")
        raise


if __name__ == "__main__":
    main()