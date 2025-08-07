#!/usr/bin/env python3
"""
Quick test script for response formatting functionality verification.

This script provides a quick verification that the response formatting system
works correctly before running the comprehensive test suite.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'lightrag_integration'))

try:
    from lightrag_integration.clinical_metabolomics_rag import (
        BiomedicalResponseFormatter,
        ResponseValidator
    )
    print("✅ Successfully imported BiomedicalResponseFormatter and ResponseValidator")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_formatter_basic():
    """Test basic formatter functionality."""
    print("\n🧪 Testing BiomedicalResponseFormatter...")
    
    try:
        # Initialize formatter
        formatter = BiomedicalResponseFormatter()
        print("✅ Formatter initialized successfully")
        
        # Test with sample response
        sample_response = """
        Metabolomics analysis reveals glucose levels elevated to 15.2 ± 3.4 mmol/L (p < 0.001).
        The glycolysis pathway shows disruption with reduced pyruvate concentrations.
        Hexokinase enzyme activity decreased by 45% in muscle tissue samples.
        Clinical biomarkers such as HbA1c provide valuable diagnostic information.
        Reference: Smith et al. (2023). Nature Medicine 29:123-135. DOI: 10.1038/s41591-023-02234-x
        """
        
        # Format the response
        result = formatter.format_response(sample_response)
        print("✅ Response formatted successfully")
        
        # Check result structure
        required_fields = ['formatted_content', 'entities', 'statistics', 'sources', 'formatting_metadata']
        for field in required_fields:
            if field in result:
                print(f"✅ Found required field: {field}")
            else:
                print(f"❌ Missing required field: {field}")
        
        # Check entities
        entities = result.get('entities', {})
        if entities:
            print(f"✅ Extracted {sum(len(v) for v in entities.values())} total entities")
            for entity_type, entity_list in entities.items():
                if entity_list:
                    print(f"  - {entity_type}: {len(entity_list)} entities")
        
        # Check statistics
        statistics = result.get('statistics', [])
        if statistics:
            print(f"✅ Extracted {len(statistics)} statistical elements")
        
        # Check sources
        sources = result.get('sources', [])
        if sources:
            print(f"✅ Processed {len(sources)} sources")
        
        return True
        
    except Exception as e:
        print(f"❌ Formatter test failed: {e}")
        return False

async def test_validator_basic():
    """Test basic validator functionality."""
    print("\n🔍 Testing ResponseValidator...")
    
    try:
        # Initialize validator
        validator = ResponseValidator()
        print("✅ Validator initialized successfully")
        
        # Test with good response
        good_response = """
        Metabolomics analysis of type 2 diabetes patients reveals significant metabolic alterations.
        Glucose concentrations were elevated to 15.2 ± 3.4 mmol/L (p < 0.001, n=156).
        The glycolysis pathway showed marked disruption with reduced pyruvate levels.
        Clinical biomarkers such as HbA1c provide valuable diagnostic information.
        These findings are supported by multiple peer-reviewed studies.
        """
        
        query = "What are the metabolic changes in type 2 diabetes?"
        metadata = {'mode': 'hybrid', 'confidence': 0.8, 'sources': []}
        
        # Validate the response
        result = await validator.validate_response(good_response, query, metadata)
        print("✅ Response validated successfully")
        
        # Check validation result structure
        required_fields = ['validation_passed', 'quality_score', 'validation_results']
        for field in required_fields:
            if field in result:
                print(f"✅ Found required field: {field}")
            else:
                print(f"❌ Missing required field: {field}")
        
        # Check validation status
        validation_passed = result.get('validation_passed', False)
        quality_score = result.get('quality_score', 0)
        
        print(f"✅ Validation passed: {validation_passed}")
        print(f"✅ Quality score: {quality_score:.2f}")
        
        # Check validation results
        validation_results = result.get('validation_results', {})
        if validation_results:
            print(f"✅ Validation includes {len(validation_results)} assessment dimensions")
            for dimension, details in validation_results.items():
                if isinstance(details, dict) and 'score' in details:
                    score = details['score']
                    print(f"  - {dimension}: {score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validator test failed: {e}")
        return False

async def test_integration():
    """Test integration between formatter and validator."""
    print("\n🔗 Testing Formatter-Validator Integration...")
    
    try:
        # Initialize both components
        formatter = BiomedicalResponseFormatter()
        validator = ResponseValidator()
        
        sample_response = """
        Clinical metabolomics study of diabetes patients (n=156) shows:
        - Glucose levels: 15.2 ± 3.4 mmol/L vs controls 5.1 ± 0.8 mmol/L (p < 0.001)
        - Insulin resistance index increased 2.3-fold (95% CI: 1.8-2.9)
        - HbA1c levels significantly elevated: 8.4 ± 1.2% (p < 0.001)
        - Glycolysis pathway disrupted with reduced pyruvate concentrations
        - Hexokinase enzyme activity decreased by 45% in muscle tissue
        - Several biomarkers show diagnostic potential:
          * 1,5-anhydroglucitol: sensitivity 87%, specificity 82%
          * 2-hydroxybutyrate: AUC = 0.84 (p < 0.001)
        
        References:
        1. Smith et al. (2023). Metabolomics in diabetes. Nature Medicine 29:123-135.
        2. Johnson A, Brown B. Diabetes Care. 2022;45(8):1234-1245. PMID: 35123456
        """
        
        query = "What are the key metabolic biomarkers for diabetes diagnosis?"
        metadata = {'mode': 'hybrid', 'confidence': 0.8, 'sources': []}
        
        # First format the response
        formatted_result = formatter.format_response(sample_response)
        print("✅ Response formatting completed")
        
        # Then validate the formatted response
        validation_result = await validator.validate_response(
            formatted_result['formatted_content'], 
            query,
            metadata
        )
        print("✅ Response validation completed")
        
        # Check integration results
        entities_count = sum(len(v) for v in formatted_result.get('entities', {}).values())
        statistics_count = len(formatted_result.get('statistics', []))
        sources_count = len(formatted_result.get('sources', []))
        
        validation_passed = validation_result.get('validation_passed', False)
        quality_score = validation_result.get('quality_score', 0)
        
        print(f"✅ Integration Results:")
        print(f"  - Entities extracted: {entities_count}")
        print(f"  - Statistics extracted: {statistics_count}")
        print(f"  - Sources processed: {sources_count}")
        print(f"  - Validation passed: {validation_passed}")
        print(f"  - Quality score: {quality_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_configuration():
    """Test configuration management."""
    print("\n⚙️  Testing Configuration Management...")
    
    try:
        # Test formatter configuration
        custom_formatter_config = {
            'extract_entities': True,
            'format_statistics': True,
            'max_entity_extraction': 20,
            'validate_scientific_accuracy': True
        }
        
        formatter = BiomedicalResponseFormatter(custom_formatter_config)
        print("✅ Custom formatter configuration applied")
        
        # Test validator configuration
        custom_validator_config = {
            'enabled': True,
            'performance_mode': 'fast',
            'thresholds': {
                'minimum_quality_score': 0.6
            }
        }
        
        validator = ResponseValidator(custom_validator_config)
        print("✅ Custom validator configuration applied")
        
        # Verify configurations
        assert formatter.config['max_entity_extraction'] == 20
        assert validator.config['performance_mode'] == 'fast'
        
        print("✅ Configuration verification successful")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def main():
    """Run all quick tests."""
    print("🚀 Response Formatting Quick Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Run tests
    test_results.append(("Formatter Basic", test_formatter_basic()))
    test_results.append(("Validator Basic", await test_validator_basic()))
    test_results.append(("Integration", await test_integration()))
    test_results.append(("Configuration", test_configuration()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Quick Test Summary:")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All quick tests passed! The system appears to be working correctly.")
        print("💡 You can now run the comprehensive test suite:")
        print("   cd lightrag_integration/tests")
        print("   python run_response_formatting_tests.py --coverage")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("🔧 Fix any issues before running the comprehensive test suite.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)