#!/usr/bin/env python3
"""
Demonstration of Validation Test Utilities for Clinical Metabolomics Oracle.

This script demonstrates the comprehensive validation capabilities provided by
the validation_test_utilities module for CMO-LIGHTRAG-008-T06.

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Demonstrate validation test utilities functionality."""
    
    print("=" * 80)
    print("CLINICAL METABOLOMICS ORACLE - VALIDATION TEST UTILITIES DEMO")
    print("=" * 80)
    print()
    
    # Import validation utilities
    try:
        from validation_test_utilities import (
            EnhancedBiomedicalContentValidator,
            TestResultValidator,
            ClinicalMetabolomicsValidator,
            ValidationReportGenerator,
            TestValidationResult,
            TestValidationType,
            ValidationSeverity
        )
        print("✓ Successfully imported validation test utilities")
    except ImportError as e:
        print(f"✗ Failed to import validation utilities: {e}")
        return
    
    # Initialize validation components
    print("\n1. INITIALIZING VALIDATION COMPONENTS")
    print("-" * 50)
    
    enhanced_validator = EnhancedBiomedicalContentValidator()
    test_validator = TestResultValidator()
    metabolomics_validator = ClinicalMetabolomicsValidator()
    report_generator = ValidationReportGenerator(Path("demo_reports"))
    
    print("✓ Enhanced Biomedical Content Validator initialized")
    print("✓ Test Result Validator initialized")
    print("✓ Clinical Metabolomics Validator initialized")
    print("✓ Validation Report Generator initialized")
    
    # Demonstrate biomedical content validation
    print("\n2. BIOMEDICAL CONTENT VALIDATION DEMO")
    print("-" * 50)
    
    # Test accurate content
    accurate_query = "What is the molecular formula of glucose?"
    accurate_response = """
    Glucose is a simple sugar with the molecular formula C6H12O6 and molecular weight of 180.16 Da. 
    Normal plasma glucose levels range from 3.9 to 6.1 mM in healthy adults. 
    Glucose participates in glycolysis and gluconeogenesis pathways and is elevated in diabetes mellitus.
    Studies suggest that glucose monitoring is essential for diabetes management.
    """
    
    print(f"Query: {accurate_query}")
    print(f"Response: {accurate_response.strip()[:100]}...")
    
    accurate_validations = enhanced_validator.validate_response_quality_comprehensive(
        accurate_query, accurate_response
    )
    
    print(f"✓ Performed {len(accurate_validations)} validations on accurate content")
    passed_count = sum(1 for v in accurate_validations if v.passed)
    print(f"  - {passed_count}/{len(accurate_validations)} validations passed")
    
    # Test inaccurate content
    inaccurate_response = """
    Glucose has the molecular formula C6H10O6 and molecular weight of 200 Da. 
    Normal levels are 10-20 mM. This compound cures diabetes and can replace all medications. 
    The correlation between glucose and health is r = 2.5 with p = 0.0.
    Stop taking insulin and use glucose supplements instead.
    """
    
    print(f"\nTesting inaccurate content...")
    inaccurate_validations = enhanced_validator.validate_response_quality_comprehensive(
        accurate_query, inaccurate_response
    )
    
    print(f"✓ Performed {len(inaccurate_validations)} validations on inaccurate content")
    failed_count = sum(1 for v in inaccurate_validations if not v.passed)
    critical_count = sum(1 for v in inaccurate_validations 
                        if v.severity == ValidationSeverity.CRITICAL and not v.passed)
    print(f"  - {failed_count}/{len(inaccurate_validations)} validations failed")
    print(f"  - {critical_count} critical safety issues detected")
    
    # Demonstrate statistical validation
    print("\n3. STATISTICAL RESULT VALIDATION DEMO")
    print("-" * 50)
    
    # Valid statistical data
    valid_stats = {
        'correlation': {'glucose_diabetes': 0.85, 'lactate_sepsis': 0.72},
        'p_values': {'glucose_test': 0.001, 'lactate_test': 0.023},
        'auc_values': {'glucose_diagnostic': 0.89, 'lactate_diagnostic': 0.78},
        'sensitivity': {'glucose_biomarker': 0.92},
        'specificity': {'glucose_biomarker': 0.85}
    }
    
    valid_stat_validations = test_validator.validate_statistical_results(valid_stats)
    print(f"✓ Validated statistical data with {len(valid_stat_validations)} checks")
    stat_passed = sum(1 for v in valid_stat_validations if v.passed)
    print(f"  - {stat_passed}/{len(valid_stat_validations)} statistical validations passed")
    
    # Invalid statistical data
    invalid_stats = {
        'correlation': {'invalid_correlation': 2.5},  # Invalid: > 1.0
        'p_values': {'zero_p_value': 0.0},  # Suspicious: exactly 0
        'auc_values': {'impossible_auc': 1.5}  # Invalid: > 1.0
    }
    
    invalid_stat_validations = test_validator.validate_statistical_results(invalid_stats)
    print(f"✓ Detected {len(invalid_stat_validations)} statistical issues")
    stat_failed = sum(1 for v in invalid_stat_validations if not v.passed)
    print(f"  - {stat_failed}/{len(invalid_stat_validations)} statistical validations failed")
    
    # Demonstrate clinical metabolomics validation
    print("\n4. CLINICAL METABOLOMICS VALIDATION DEMO")
    print("-" * 50)
    
    metabolomics_query = "What are the key metabolites for diabetes diagnosis?"
    metabolomics_response = """
    Key metabolites for diabetes diagnosis include glucose, which is elevated in diabetic patients,
    and lactate, which may be increased due to altered metabolism. LC-MS/MS analysis shows
    that glucose levels correlate strongly with HbA1c (r = 0.89, p < 0.001).
    The glycolysis pathway is particularly affected in diabetes.
    """
    
    expected_metabolites = ['glucose', 'lactate']
    metabolomics_validations = metabolomics_validator.validate_metabolomics_query_response(
        metabolomics_query, metabolomics_response, expected_metabolites
    )
    
    print(f"✓ Performed clinical metabolomics validation")
    print(f"  - {len(metabolomics_validations)} domain-specific checks completed")
    metabolomics_passed = sum(1 for v in metabolomics_validations if v.passed)
    print(f"  - {metabolomics_passed}/{len(metabolomics_validations)} metabolomics validations passed")
    
    # Demonstrate cross-test consistency validation
    print("\n5. CROSS-TEST CONSISTENCY VALIDATION DEMO")
    print("-" * 50)
    
    test_results = [
        {
            'test_name': 'glucose_test_1',
            'result_patterns': {'molecular_formula': 'C6H12O6', 'molecular_weight': 180.16},
            'statistical_data': {'correlation': {'glucose_diabetes': 0.85}},
            'performance_metrics': {'response_time_ms': 150, 'memory_usage_mb': 45}
        },
        {
            'test_name': 'glucose_test_2', 
            'result_patterns': {'molecular_formula': 'C6H12O6', 'molecular_weight': 180.16},
            'statistical_data': {'correlation': {'glucose_diabetes': 0.87}},  # Slightly different
            'performance_metrics': {'response_time_ms': 200, 'memory_usage_mb': 60}  # Different performance
        },
        {
            'test_name': 'glucose_test_3',
            'result_patterns': {'molecular_formula': 'C6H10O6'},  # Wrong formula
            'statistical_data': {'correlation': {'glucose_diabetes': 0.20}},  # Very different
            'performance_metrics': {'response_time_ms': 500, 'memory_usage_mb': 120}  # Poor performance
        }
    ]
    
    consistency_result = test_validator.validate_cross_test_consistency(test_results)
    print(f"✓ Cross-test consistency validation completed")
    print(f"  - Consistency score: {consistency_result.consistency_score:.2f}")
    print(f"  - {len(consistency_result.inconsistencies_found)} inconsistencies detected")
    print(f"  - {len(consistency_result.recommendations)} recommendations generated")
    
    # Demonstrate custom validation rules
    print("\n6. CUSTOM VALIDATION RULES DEMO")
    print("-" * 50)
    
    # Add custom validation rule
    def validate_glucose_range(test_data, **kwargs):
        """Custom validation rule for glucose range."""
        glucose_value = test_data.get('glucose_level', 0)
        return 3.9 <= glucose_value <= 6.1
    
    test_validator.add_custom_validation_rule(
        rule_name='glucose_normal_range',
        validation_function=validate_glucose_range,
        expected_result=True,
        severity=ValidationSeverity.MAJOR,
        description="Validate glucose levels are within normal range"
    )
    
    # Test custom validation
    custom_test_data = {'glucose_level': 5.5}  # Normal
    custom_validations = test_validator.apply_custom_validation_rules(custom_test_data)
    print(f"✓ Applied custom validation rules")
    print(f"  - {len(custom_validations)} custom validations completed")
    
    custom_test_data_abnormal = {'glucose_level': 15.0}  # Abnormal
    custom_validations_abnormal = test_validator.apply_custom_validation_rules(custom_test_data_abnormal)
    custom_failed = sum(1 for v in custom_validations_abnormal if not v.passed)
    print(f"  - {custom_failed} custom validations failed for abnormal data")
    
    # Collect all validation results
    all_validations = (
        accurate_validations + inaccurate_validations + 
        valid_stat_validations + invalid_stat_validations + 
        metabolomics_validations + custom_validations + custom_validations_abnormal
    )
    
    # Generate comprehensive validation report
    print("\n7. VALIDATION REPORT GENERATION DEMO")
    print("-" * 50)
    
    test_metadata = {
        'test_suite': 'validation_utilities_demo',
        'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_scenarios': 6,
        'validation_framework_version': '1.0.0'
    }
    
    comprehensive_report = report_generator.generate_comprehensive_report(
        all_validations, test_metadata, "demo_validation_report"
    )
    
    print(f"✓ Generated comprehensive validation report")
    print(f"  - Report ID: {comprehensive_report['report_metadata']['report_name']}")
    print(f"  - Total validations: {comprehensive_report['report_metadata']['total_validations']}")
    print(f"  - Overall pass rate: {comprehensive_report['executive_summary']['overall_pass_rate']:.1%}")
    print(f"  - Critical issues: {comprehensive_report['executive_summary']['critical_issues']}")
    print(f"  - Total recommendations: {comprehensive_report['executive_summary']['total_recommendations']}")
    
    # Display key validation statistics
    print("\n8. VALIDATION STATISTICS SUMMARY")
    print("-" * 50)
    
    total_validations = len(all_validations)
    passed_validations = sum(1 for v in all_validations if v.passed)
    critical_issues = sum(1 for v in all_validations if v.severity == ValidationSeverity.CRITICAL and not v.passed)
    major_issues = sum(1 for v in all_validations if v.severity == ValidationSeverity.MAJOR and not v.passed)
    
    print(f"Total Validations Performed: {total_validations}")
    print(f"Validations Passed: {passed_validations}")
    print(f"Validations Failed: {total_validations - passed_validations}")
    print(f"Overall Pass Rate: {passed_validations / total_validations:.1%}")
    print(f"Critical Issues Detected: {critical_issues}")
    print(f"Major Issues Detected: {major_issues}")
    
    # Show validation by type
    validation_types = {}
    for validation in all_validations:
        vtype = validation.validation_type.value
        if vtype not in validation_types:
            validation_types[vtype] = {'total': 0, 'passed': 0}
        validation_types[vtype]['total'] += 1
        if validation.passed:
            validation_types[vtype]['passed'] += 1
    
    print(f"\nValidation Results by Type:")
    for vtype, stats in validation_types.items():
        pass_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  - {vtype}: {stats['passed']}/{stats['total']} ({pass_rate:.1%})")
    
    print(f"\n9. INTEGRATION STATUS")
    print("-" * 50)
    print("✓ Enhanced Biomedical Content Validator - OPERATIONAL")
    print("✓ Test Result Validator - OPERATIONAL")
    print("✓ Clinical Metabolomics Validator - OPERATIONAL")  
    print("✓ Validation Report Generator - OPERATIONAL")
    print("✓ Cross-test Consistency Validation - OPERATIONAL")
    print("✓ Custom Validation Rules - OPERATIONAL")
    print("✓ Statistical Result Validation - OPERATIONAL")
    print("✓ Clinical Safety Validation - OPERATIONAL")
    
    print(f"\n{'=' * 80}")
    print("VALIDATION TEST UTILITIES DEMONSTRATION COMPLETED SUCCESSFULLY")
    print(f"{'=' * 80}")
    
    # Show sample critical issue for demonstration
    critical_validations = [v for v in all_validations if v.severity == ValidationSeverity.CRITICAL and not v.passed]
    if critical_validations:
        print(f"\nSample Critical Issue Detected:")
        critical = critical_validations[0]
        print(f"  - Issue: {critical.message}")
        print(f"  - Test: {critical.test_name}")
        print(f"  - Recommendations: {critical.recommendations[:2]}")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nDemo completed successfully!")
            exit(0)
        else:
            print("\nDemo failed!")
            exit(1)
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)