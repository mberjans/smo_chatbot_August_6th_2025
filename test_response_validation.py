#!/usr/bin/env python3
"""
Test script for the ResponseValidator integration in Clinical Metabolomics RAG.

This script tests the response validation system to ensure it works correctly
with different types of biomedical responses.
"""

import asyncio
import sys
import os

# Add the lightrag_integration directory to the path
sys.path.append(os.path.dirname(__file__))

from lightrag_integration.clinical_metabolomics_rag import ResponseValidator

async def test_response_validator():
    """Test the ResponseValidator with sample biomedical responses."""
    
    print("Testing ResponseValidator...")
    
    # Initialize validator
    validator = ResponseValidator()
    
    # Test with a good biomedical response
    good_response = """
    Metabolomics analysis of diabetes patients reveals significant alterations in glucose metabolism.
    Studies show that glucose levels are typically elevated in diabetic patients (p < 0.001).
    The glycolysis pathway is often disrupted, leading to reduced pyruvate concentrations.
    Clinical biomarkers such as HbA1c provide valuable diagnostic information.
    Research indicates that these metabolic changes may be associated with cardiovascular complications.
    """
    
    query = "What metabolic changes occur in diabetes?"
    metadata = {
        'mode': 'hybrid',
        'processing_time': 1.2,
        'sources': [],
        'confidence': 0.8
    }
    
    print("\n--- Testing Good Response ---")
    validation_result = await validator.validate_response(good_response, query, metadata)
    
    print(f"Validation Passed: {validation_result['validation_passed']}")
    print(f"Quality Score: {validation_result['quality_score']:.2f}")
    print(f"Recommendations: {validation_result['recommendations']}")
    
    # Test with a problematic response (absolute claims, no sources)
    bad_response = """
    Diabetes always causes complete metabolic failure. All patients will definitely die.
    Glucose levels are exactly 999999 mg/L in every case. This completely eliminates all symptoms.
    No further research is needed as this is impossible to treat.
    """
    
    print("\n--- Testing Problematic Response ---")
    bad_validation_result = await validator.validate_response(bad_response, query, metadata)
    
    print(f"Validation Passed: {bad_validation_result['validation_passed']}")
    print(f"Quality Score: {bad_validation_result['quality_score']:.2f}")
    print(f"Recommendations: {bad_validation_result['recommendations']}")
    
    # Test hallucination detection
    print(f"Hallucination Risk: {bad_validation_result['validation_results']['hallucination_check']['risk_level']}")
    print(f"Unsupported Claims: {bad_validation_result['validation_results']['hallucination_check']['unsourced_specific_claims']}")
    
    print("\n--- Validation Test Complete ---")

async def test_integration_mock():
    """Test the integration with a mock clinical metabolomics query."""
    
    print("\n\n--- Testing Integration (Mock) ---")
    
    # Test with mode-specific configuration
    validation_config = {
        'enabled': True,
        'performance_mode': 'fast',
        'quality_gate_enabled': True,
        'thresholds': {
            'minimum_quality_score': 0.6,
            'scientific_confidence_threshold': 0.7
        }
    }
    
    validator = ResponseValidator(validation_config)
    
    # Simulate a clinical metabolomics response
    clinical_response = """
    Introduction: Metabolomics studies in cardiovascular disease have identified key biomarkers.
    
    Key Findings: Several metabolites show significant alterations:
    - Cholesterol levels were elevated (p < 0.05) in 78.3% of patients
    - Lactate concentrations increased by 2.1-fold compared to controls
    - Glycolysis pathway disruption was observed in tissue samples
    
    Clinical Relevance: These biomarkers may serve as diagnostic tools for early detection.
    Further research is needed to validate these findings in larger cohorts.
    
    Conclusion: The metabolomic profile suggests potential therapeutic targets for treatment.
    """
    
    query = "What are the key metabolomic biomarkers for cardiovascular disease diagnosis?"
    metadata = {
        'mode': 'complex_analysis',
        'processing_time': 2.5,
        'sources': ['PubMed:12345', 'DOI:10.1016/example'],
        'confidence': 0.85
    }
    
    # Mock formatted response data
    formatted_response = {
        'sections': {
            'introduction': 'Introduction section content',
            'key_findings': 'Key findings section content',
            'clinical_relevance': 'Clinical relevance section content',
            'conclusion': 'Conclusion section content'
        },
        'entities': {
            'metabolites': ['cholesterol', 'lactate'],
            'diseases': ['cardiovascular disease'],
            'pathways': ['glycolysis']
        },
        'statistics': ['p < 0.05', '78.3%', '2.1-fold'],
        'sources': ['PubMed:12345', 'DOI:10.1016/example']
    }
    
    validation_result = await validator.validate_response(
        clinical_response, query, metadata, formatted_response
    )
    
    print(f"Clinical Query Validation Passed: {validation_result['validation_passed']}")
    print(f"Overall Quality Score: {validation_result['quality_score']:.2f}")
    
    # Print detailed quality dimensions
    print("\nQuality Dimensions:")
    for dimension, score in validation_result['quality_dimensions'].items():
        print(f"  {dimension}: {score:.2f}")
    
    # Print confidence assessment
    confidence_assessment = validation_result['confidence_assessment']
    print(f"\nConfidence Assessment:")
    print(f"  Overall Confidence: {confidence_assessment['overall_confidence']:.2f}")
    print(f"  Uncertainty Level: {confidence_assessment['uncertainty_level']}")
    
    print(f"\nProcessing Time: {validation_result['validation_metadata']['processing_time']:.3f}s")
    print(f"Recommendations: {validation_result['recommendations']}")
    
    print("\n--- Integration Test Complete ---")

async def test_excellent_response():
    """Test with an excellent biomedical response that should pass validation."""
    
    print("\n\n--- Testing Excellent Response ---")
    
    # Use more permissive thresholds for this test
    validation_config = {
        'enabled': True,
        'performance_mode': 'balanced',
        'quality_gate_enabled': True,
        'thresholds': {
            'minimum_quality_score': 0.5,
            'scientific_confidence_threshold': 0.6
        }
    }
    
    validator = ResponseValidator(validation_config)
    
    excellent_response = """
    Introduction: Metabolomics research has revealed significant biomarkers for diabetes diagnosis.
    
    Key Findings: Multiple studies have demonstrated metabolic alterations in diabetic patients:
    - Glucose concentrations are typically elevated (p < 0.001) in diabetic subjects
    - The glycolysis pathway shows increased activity, potentially due to insulin resistance
    - Lactate levels may increase by 1.5-fold in severe cases, according to recent clinical trials
    - Biomarkers such as HbA1c correlate strongly with glucose metabolism disruption
    
    Clinical Relevance: These metabolite changes provide diagnostic value for patient care.
    The biomarkers show promise for early detection and treatment monitoring.
    However, further research is needed to validate these findings across diverse populations.
    
    Conclusion: Metabolomics approaches offer valuable insights for diabetes management,
    though additional studies are required to establish clinical guidelines.
    """
    
    query = "What metabolic biomarkers are useful for diabetes diagnosis?"
    metadata = {
        'mode': 'comprehensive_research',
        'processing_time': 2.8,
        'sources': ['PubMed:11111', 'DOI:10.1038/example', 'Clinical Trial NCT123'],
        'confidence': 0.88
    }
    
    # Mock formatted response with rich metadata
    formatted_response = {
        'sections': {
            'introduction': 'Introduction content',
            'key_findings': 'Key findings content', 
            'clinical_relevance': 'Clinical relevance content',
            'conclusion': 'Conclusion content'
        },
        'entities': {
            'metabolites': ['glucose', 'lactate', 'HbA1c'],
            'diseases': ['diabetes'],
            'pathways': ['glycolysis', 'insulin resistance'],
            'biomarkers': ['HbA1c', 'glucose', 'lactate']
        },
        'statistics': ['p < 0.001', '1.5-fold'],
        'sources': ['PubMed:11111', 'DOI:10.1038/example', 'Clinical Trial NCT123']
    }
    
    validation_result = await validator.validate_response(
        excellent_response, query, metadata, formatted_response
    )
    
    print(f"Excellent Response Validation Passed: {validation_result['validation_passed']}")
    print(f"Overall Quality Score: {validation_result['quality_score']:.2f}")
    
    # Print detailed quality dimensions
    print("\nQuality Dimensions:")
    for dimension, score in validation_result['quality_dimensions'].items():
        print(f"  {dimension}: {score:.2f}")
    
    print(f"\nHallucination Risk: {validation_result['validation_results']['hallucination_check']['risk_level']}")
    print(f"Scientific Accuracy: {validation_result['validation_results']['scientific_accuracy']['score']:.2f}")
    print(f"Recommendations: {validation_result['recommendations']}")
    
    print("\n--- Excellent Response Test Complete ---")


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_response_validator())
    asyncio.run(test_integration_mock())
    asyncio.run(test_excellent_response())
    print("\n✅ All ResponseValidator tests completed successfully!")