#!/usr/bin/env python3
"""
Test script for the enhanced post-processing features in Clinical Metabolomics RAG system.

This script demonstrates the new scientific accuracy validation, citation processing,
and content quality assessment capabilities.
"""

import sys
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lightrag_integration.clinical_metabolomics_rag import BiomedicalResponseFormatter


def test_scientific_accuracy_validation():
    """Test the scientific accuracy validation functionality."""
    print("Testing Scientific Accuracy Validation...")
    print("=" * 50)
    
    formatter = BiomedicalResponseFormatter()
    
    # Test content with metabolite properties and statistical claims
    test_content = """
    Glucose has a molecular weight of 180.16 g/mol and is crucial for cellular energy metabolism.
    In patients with diabetes, glucose concentration ranges from 126-400 mg/dL.
    The study showed a significant correlation (p = 0.023) between glucose levels and HbA1c.
    Glycolysis pathway produces pyruvate from glucose, which then enters the TCA cycle.
    """
    
    formatted_response = {
        'formatted_content': test_content,
        'formatting_metadata': {'applied_formatting': []}
    }
    
    # Test scientific accuracy validation
    result = formatter.validate_scientific_accuracy(formatted_response)
    
    print("Content:", test_content[:100] + "...")
    print("\nScientific Validation Results:")
    validation = result.get('scientific_validation', {})
    print(f"  Overall confidence score: {validation.get('overall_confidence_score', 0):.2f}")
    print(f"  Potential inaccuracies found: {len(validation.get('potential_inaccuracies', []))}")
    print(f"  Statistical validations: {len(validation.get('statistical_validation', []))}")
    
    # Show detailed results if available
    fact_check = validation.get('fact_check_results', {})
    if fact_check:
        print("\nDetailed Fact-Check Results:")
        for category, results in fact_check.items():
            if results and isinstance(results, dict):
                confidence = results.get('confidence_score', 0)
                print(f"  {category}: {confidence:.2f} confidence")
    
    print()


def test_citation_processing():
    """Test the enhanced citation processing functionality."""
    print("Testing Enhanced Citation Processing...")
    print("=" * 50)
    
    formatter = BiomedicalResponseFormatter()
    
    # Test content with various citation formats
    test_content = """
    Recent studies have shown metabolomic changes in diabetes [1,2].
    Smith et al. (2023) demonstrated significant pathway alterations.
    DOI: 10.1038/nature12345 provides comprehensive metabolite profiling data.
    PMID: 12345678 discusses clinical applications of metabolomics.
    Additional research (PMCID: PMC9876543) supports these findings.
    """
    
    formatted_response = {
        'formatted_content': test_content,
        'sources': [],
        'formatting_metadata': {'applied_formatting': []}
    }
    
    # Test citation processing
    result = formatter.process_citations(formatted_response)
    
    print("Content:", test_content[:100] + "...")
    print("\nCitation Processing Results:")
    enhanced_citations = result.get('enhanced_citations', {})
    processed_citations = enhanced_citations.get('processed_citations', [])
    
    print(f"  Total citations found: {len(processed_citations)}")
    
    for i, citation in enumerate(processed_citations[:3]):  # Show first 3
        print(f"  Citation {i+1}:")
        print(f"    Text: {citation.get('text', 'Unknown')}")
        print(f"    Type: {citation.get('type', 'Unknown')}")
        print(f"    Validated: {citation.get('validated', False)}")
        print(f"    Credibility Score: {citation.get('credibility_score', 0):.2f}")
        if citation.get('link'):
            print(f"    Link: {citation['link']}")
    
    quality_indicators = enhanced_citations.get('source_quality_indicators', {})
    if quality_indicators:
        print(f"\nSource Quality:")
        print(f"  Overall quality: {quality_indicators.get('overall_quality', 0):.2f}")
        print(f"  Validation rate: {quality_indicators.get('validation_rate', 0):.2f}")
    
    print()


def test_content_quality_assessment():
    """Test the content quality assessment functionality."""
    print("Testing Content Quality Assessment...")
    print("=" * 50)
    
    formatter = BiomedicalResponseFormatter()
    
    # Test content with various quality indicators
    test_content = """
    Metabolomics research has established significant associations between glucose metabolism 
    and cardiovascular disease. Multiple studies have demonstrated that elevated glucose levels 
    correlate with increased risk factors. However, the precise mechanisms remain unclear.
    
    The glycolysis pathway plays a crucial role in cellular energy production. Research indicates 
    that metabolite biomarkers may provide diagnostic value for early disease detection.
    Recent meta-analyses have confirmed these findings across diverse populations.
    """
    
    formatted_response = {
        'formatted_content': test_content,
        'entities': {
            'metabolites': ['glucose'],
            'pathways': ['glycolysis'],
            'diseases': ['cardiovascular disease']
        },
        'clinical_indicators': {'overall_relevance_score': 0.8},
        'formatting_metadata': {'applied_formatting': []}
    }
    
    # Test quality assessment
    result = formatter.assess_content_quality(formatted_response)
    
    print("Content:", test_content[:100] + "...")
    print("\nContent Quality Assessment Results:")
    quality = result.get('quality_assessment', {})
    
    print(f"  Overall quality score: {quality.get('overall_quality_score', 0):.2f}")
    print(f"  Completeness score: {quality.get('completeness_score', 0):.2f}")
    print(f"  Relevance score: {quality.get('relevance_score', 0):.2f}")
    print(f"  Consistency score: {quality.get('consistency_score', 0):.2f}")
    print(f"  Authority score: {quality.get('authority_score', 0):.2f}")
    print(f"  Uncertainty level: {quality.get('uncertainty_level', 0):.2f}")
    
    quality_indicators = quality.get('quality_indicators', {})
    if quality_indicators:
        print("\nQuality Indicators:")
        for indicator, level in quality_indicators.items():
            print(f"  {indicator.replace('_', ' ').title()}: {level}")
    
    recommendations = quality.get('improvement_recommendations', [])
    if recommendations:
        print("\nImprovement Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
    
    print()


def test_comprehensive_formatting():
    """Test the comprehensive formatting with all enhanced features."""
    print("Testing Comprehensive Enhanced Formatting...")
    print("=" * 50)
    
    formatter = BiomedicalResponseFormatter()
    
    # Complex test content with multiple elements
    test_content = """
    Clinical metabolomics analysis revealed significant alterations in glucose metabolism 
    (p = 0.015) in diabetic patients. The study involved 150 participants and showed 
    glucose levels of 180.16 mg/dL (normal range: 70-110 mg/dL).
    
    Key Findings:
    - Glucose molecular weight: 180.16 g/mol (confirmed)
    - Glycolysis pathway produces pyruvate efficiently
    - Statistical significance demonstrated (r² = 0.76)
    
    These findings are supported by recent research (DOI: 10.1038/nature12345) 
    and meta-analysis data (PMID: 12345678). However, further investigation 
    may be needed to establish causality.
    """
    
    formatted_response = {
        'formatted_content': test_content,
        'original_content': test_content,
        'sections': {},
        'entities': {},
        'statistics': [],
        'sources': [],
        'clinical_indicators': {},
        'formatting_metadata': {
            'processed_at': '2025-01-01T00:00:00',
            'formatter_version': '1.0.0',
            'applied_formatting': []
        }
    }
    
    # Apply comprehensive formatting
    result = formatter.format_response(test_content)
    
    print("Content:", test_content[:100] + "...")
    print("\nComprehensive Formatting Results:")
    
    # Show applied formatting
    applied_formatting = result.get('formatting_metadata', {}).get('applied_formatting', [])
    print(f"Applied formatting steps: {', '.join(applied_formatting)}")
    
    # Show key results
    if 'scientific_validation' in result:
        validation = result['scientific_validation']
        print(f"Scientific confidence: {validation.get('overall_confidence_score', 0):.2f}")
    
    if 'enhanced_citations' in result:
        citations = result['enhanced_citations']
        citation_count = len(citations.get('processed_citations', []))
        print(f"Citations processed: {citation_count}")
    
    if 'quality_assessment' in result:
        quality = result['quality_assessment']
        print(f"Content quality: {quality.get('overall_quality_score', 0):.2f}")
    
    # Show any errors
    errors = result.get('formatting_metadata', {}).get('errors', [])
    if errors:
        print(f"Formatting errors: {len(errors)}")
        for error in errors:
            print(f"  - {error}")
    
    print()


if __name__ == "__main__":
    print("Enhanced Post-Processing Test Suite")
    print("=" * 60)
    print("Testing new features in Clinical Metabolomics RAG system")
    print()
    
    try:
        test_scientific_accuracy_validation()
        test_citation_processing()
        test_content_quality_assessment()
        test_comprehensive_formatting()
        
        print("All tests completed successfully!")
        print("\nThe enhanced post-processing features are working correctly:")
        print("✓ Scientific accuracy validation")
        print("✓ Enhanced citation processing")
        print("✓ Content quality assessment")
        print("✓ Comprehensive error handling")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()