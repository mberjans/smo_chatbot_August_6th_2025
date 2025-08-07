#!/usr/bin/env python3
"""
Validation Test Utilities for Clinical Metabolomics Oracle - CMO-LIGHTRAG-008-T06.

This module provides comprehensive test result validation and assertion helpers specifically
designed for biomedical content validation in the Clinical Metabolomics Oracle system.

Key Components:
1. BiomedicalContentValidator: Enhanced response quality assessment and medical terminology validation
2. TestResultValidator: Statistical result validation and cross-test consistency checking
3. ClinicalMetabolomicsValidator: Domain-specific validation for clinical metabolomics
4. ValidationReportGenerator: Detailed diagnostics and validation reporting
5. Integration with existing TestEnvironmentManager, MockSystemFactory, PerformanceAssertionHelper

This implementation focuses on:
- Response quality assessment with clinical accuracy verification
- Cross-document consistency validation
- Statistical result pattern validation
- Custom validation rules for metabolomics domain
- Comprehensive validation reporting with actionable diagnostics

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
License: Clinical Metabolomics Oracle Project License
"""

import pytest
import asyncio
import time
import json
import re
import statistics
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Pattern
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta

# Import from existing test infrastructure
from test_utilities import TestEnvironmentManager, MockSystemFactory, SystemComponent
from performance_test_utilities import PerformanceAssertionHelper
from biomedical_test_fixtures import ClinicalMetabolomicsDataGenerator, MetaboliteData
from validation_fixtures import (
    BiomedicalContentValidator, ValidationResult, ValidationReport, 
    ValidationLevel, ValidationType, ResponseQualityAssessor
)


# =====================================================================
# ENHANCED VALIDATION TYPES AND LEVELS
# =====================================================================

class TestValidationType(Enum):
    """Extended validation types for test result validation."""
    RESPONSE_QUALITY = "response_quality"
    STATISTICAL_CONSISTENCY = "statistical_consistency"
    CROSS_TEST_CONSISTENCY = "cross_test_consistency"
    PATTERN_VALIDATION = "pattern_validation"
    DOMAIN_ACCURACY = "domain_accuracy"
    COMPARATIVE_VALIDATION = "comparative_validation"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    DATA_INTEGRITY = "data_integrity"
    PERFORMANCE_VALIDATION = "performance_validation"
    INTEGRATION_VALIDATION = "integration_validation"


class ValidationSeverity(Enum):
    """Validation severity levels for test assertions."""
    BLOCKER = "blocker"          # Test must fail
    CRITICAL = "critical"        # High priority issue
    MAJOR = "major"             # Significant concern
    MINOR = "minor"             # Low priority issue
    INFO = "info"               # Informational
    WARNING = "warning"         # Potential concern


@dataclass
class TestValidationResult:
    """Enhanced validation result for test scenarios."""
    validation_id: str
    test_name: str
    validation_type: TestValidationType
    severity: ValidationSeverity
    passed: bool
    confidence: float
    message: str
    actual_value: Any = None
    expected_value: Any = None
    tolerance: float = 0.0
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'validation_id': self.validation_id,
            'test_name': self.test_name,
            'validation_type': self.validation_type.value,
            'severity': self.severity.value,
            'passed': self.passed,
            'confidence': self.confidence,
            'message': self.message,
            'actual_value': self.actual_value,
            'expected_value': self.expected_value,
            'tolerance': self.tolerance,
            'evidence': self.evidence,
            'recommendations': self.recommendations,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }


@dataclass
class ValidationAssertion:
    """Test assertion for validation."""
    assertion_name: str
    validation_function: Callable
    expected_result: bool
    severity: ValidationSeverity
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossTestValidationResult:
    """Result of cross-test validation."""
    validation_suite: str
    tests_compared: List[str]
    consistency_score: float
    inconsistencies_found: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: float = field(default_factory=time.time)


# =====================================================================
# ENHANCED BIOMEDICAL CONTENT VALIDATOR
# =====================================================================

class EnhancedBiomedicalContentValidator(BiomedicalContentValidator):
    """
    Enhanced biomedical content validator with advanced response quality assessment
    and clinical accuracy verification capabilities.
    """
    
    def __init__(self):
        super().__init__()
        self.quality_assessor = ResponseQualityAssessor()
        self.clinical_terminology = self._load_clinical_terminology()
        self.validation_patterns = self._load_validation_patterns()
        self.cross_reference_cache = {}
        self.logger = logging.getLogger(__name__)
        
    def _load_clinical_terminology(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive clinical terminology database."""
        return {
            'metabolomics_terms': {
                'lc_ms_ms': {
                    'full_name': 'Liquid Chromatography Tandem Mass Spectrometry',
                    'abbreviations': ['LC-MS/MS', 'LC-MS²', 'LCMSMS'],
                    'context': 'analytical_technique',
                    'synonyms': ['liquid chromatography mass spectrometry'],
                    'applications': ['metabolite_identification', 'quantification']
                },
                'biomarker': {
                    'definition': 'measurable biological characteristic indicating disease state',
                    'types': ['diagnostic', 'prognostic', 'predictive', 'monitoring'],
                    'context': 'clinical_diagnostic',
                    'validation_requirements': ['sensitivity', 'specificity', 'clinical_utility']
                },
                'metabolic_pathway': {
                    'definition': 'series of chemical reactions in living cells',
                    'categories': ['catabolic', 'anabolic', 'amphibolic'],
                    'regulation': ['allosteric', 'covalent_modification', 'transcriptional'],
                    'clinical_relevance': ['disease_mechanisms', 'therapeutic_targets']
                }
            },
            'clinical_units': {
                'concentration': {
                    'molar': ['M', 'mM', 'μM', 'nM', 'pM'],
                    'mass': ['mg/dL', 'g/L', 'μg/mL', 'ng/mL', 'pg/mL'],
                    'conversions': {
                        'glucose_mg_dl_to_mm': lambda x: x / 18.0,
                        'cholesterol_mg_dl_to_mm': lambda x: x / 38.7
                    }
                }
            }
        }
    
    def _load_validation_patterns(self) -> Dict[str, List[Pattern]]:
        """Load regex patterns for content validation."""
        return {
            'clinical_values': [
                re.compile(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*(mg/dL|mM|μM|g/L)'),
                re.compile(r'normal\s+range[:\s]*(\d+\.?\d*)\s*to\s*(\d+\.?\d*)'),
                re.compile(r'reference\s+values?[:\s]*(\d+\.?\d*)[±\-~](\d+\.?\d*)')
            ],
            'statistical_values': [
                re.compile(r'p\s*[<>=]\s*(\d*\.?\d+)'),
                re.compile(r'r\s*=\s*([+-]?\d*\.?\d+)'),
                re.compile(r'auc\s*[=:]\s*(\d*\.?\d+)'),
                re.compile(r'sensitivity\s*[=:]\s*(\d+\.?\d*)%?'),
                re.compile(r'specificity\s*[=:]\s*(\d+\.?\d*)%?')
            ],
            'temporal_expressions': [
                re.compile(r'(\d+)\s*(hours?|hrs?|days?|weeks?|months?|years?)'),
                re.compile(r'(baseline|follow-up|pre-treatment|post-treatment)'),
                re.compile(r'(before|after|during)\s+treatment')
            ]
        }
    
    def validate_response_quality_comprehensive(self, 
                                              query: str, 
                                              response: str,
                                              context: Optional[Dict[str, Any]] = None) -> List[TestValidationResult]:
        """Perform comprehensive response quality validation."""
        validations = []
        
        # Basic biomedical content validation
        basic_validation_report = self.validate_content(response, context)
        
        # Convert basic validations to test validation results
        for validation in basic_validation_report.validation_results:
            test_validation = TestValidationResult(
                validation_id=f"content_{validation.validation_id}",
                test_name="biomedical_content_validation",
                validation_type=TestValidationType.RESPONSE_QUALITY,
                severity=self._map_validation_level_to_severity(validation.validation_level),
                passed=validation.passed,
                confidence=validation.confidence,
                message=validation.message,
                evidence=validation.evidence,
                recommendations=validation.suggestions,
                metadata={'original_validation_type': validation.validation_type.value}
            )
            validations.append(test_validation)
        
        # Advanced quality assessment
        quality_assessment = self.quality_assessor.assess_response_quality(
            query, response, basic_validation_report
        )
        
        # Validate response completeness
        completeness_validation = self._validate_response_completeness(
            query, response, quality_assessment
        )
        validations.extend(completeness_validation)
        
        # Validate clinical accuracy
        clinical_accuracy_validation = self._validate_clinical_accuracy(
            response, context or {}
        )
        validations.extend(clinical_accuracy_validation)
        
        # Validate terminology consistency
        terminology_validation = self._validate_advanced_terminology(response)
        validations.extend(terminology_validation)
        
        return validations
    
    def validate_cross_document_consistency(self, 
                                          responses: List[Dict[str, str]],
                                          consistency_criteria: Optional[Dict[str, Any]] = None) -> List[TestValidationResult]:
        """Validate consistency across multiple document responses."""
        validations = []
        
        if len(responses) < 2:
            return [TestValidationResult(
                validation_id=f"cross_doc_insufficient_{int(time.time())}",
                test_name="cross_document_consistency",
                validation_type=TestValidationType.CROSS_TEST_CONSISTENCY,
                severity=ValidationSeverity.WARNING,
                passed=False,
                confidence=1.0,
                message="Insufficient responses for cross-document consistency validation",
                recommendations=["Provide at least 2 responses for consistency validation"]
            )]
        
        # Extract key information from each response
        response_data = []
        for i, response_item in enumerate(responses):
            query = response_item.get('query', '')
            response = response_item.get('response', '')
            
            # Extract metabolite information
            metabolites = self._extract_metabolites_from_content(response)
            
            # Extract numerical values
            numerical_values = self._extract_numerical_values(response)
            
            # Extract clinical claims
            clinical_claims = self._extract_clinical_claims(response)
            
            response_data.append({
                'index': i,
                'query': query,
                'response': response,
                'metabolites': metabolites,
                'numerical_values': numerical_values,
                'clinical_claims': clinical_claims
            })
        
        # Check for consistency in metabolite information
        metabolite_consistency = self._validate_metabolite_consistency(response_data)
        validations.extend(metabolite_consistency)
        
        # Check for consistency in numerical values
        numerical_consistency = self._validate_numerical_consistency(response_data)
        validations.extend(numerical_consistency)
        
        # Check for consistency in clinical claims
        clinical_consistency = self._validate_clinical_claim_consistency(response_data)
        validations.extend(clinical_consistency)
        
        return validations
    
    def validate_temporal_consistency(self, 
                                    responses: List[Dict[str, Any]],
                                    time_threshold_hours: float = 24.0) -> List[TestValidationResult]:
        """Validate temporal consistency in responses."""
        validations = []
        
        # Sort responses by timestamp
        sorted_responses = sorted(responses, key=lambda x: x.get('timestamp', 0))
        
        # Check for temporal drift in responses
        for i in range(1, len(sorted_responses)):
            current_response = sorted_responses[i]
            previous_response = sorted_responses[i-1]
            
            time_diff = current_response.get('timestamp', 0) - previous_response.get('timestamp', 0)
            
            if time_diff <= time_threshold_hours * 3600:  # Convert to seconds
                # Responses are close in time, should be consistent
                consistency_score = self._calculate_response_similarity(
                    previous_response.get('response', ''),
                    current_response.get('response', '')
                )
                
                if consistency_score < 0.7:  # Below 70% similarity
                    validations.append(TestValidationResult(
                        validation_id=f"temporal_inconsistency_{int(time.time())}",
                        test_name="temporal_consistency",
                        validation_type=TestValidationType.TEMPORAL_CONSISTENCY,
                        severity=ValidationSeverity.MAJOR,
                        passed=False,
                        confidence=0.8,
                        message=f"Temporal inconsistency detected: {consistency_score:.2f} similarity",
                        actual_value=consistency_score,
                        expected_value=0.7,
                        metadata={
                            'time_diff_hours': time_diff / 3600,
                            'response_indices': [i-1, i]
                        },
                        recommendations=[
                            "Review responses for consistency in temporal window",
                            "Consider response caching for similar queries"
                        ]
                    ))
        
        return validations
    
    def _validate_response_completeness(self, 
                                      query: str, 
                                      response: str, 
                                      quality_assessment: Dict[str, Any]) -> List[TestValidationResult]:
        """Validate response completeness based on query requirements."""
        validations = []
        
        completeness_score = quality_assessment['metrics']['completeness']
        
        if completeness_score < 0.7:
            validations.append(TestValidationResult(
                validation_id=f"completeness_{int(time.time())}",
                test_name="response_completeness",
                validation_type=TestValidationType.RESPONSE_QUALITY,
                severity=ValidationSeverity.MAJOR,
                passed=False,
                confidence=0.8,
                message=f"Response completeness below threshold: {completeness_score:.2f}",
                actual_value=completeness_score,
                expected_value=0.7,
                recommendations=[
                    "Address all aspects of the query",
                    "Provide more comprehensive information",
                    "Include relevant clinical context"
                ]
            ))
        
        # Check for specific completeness criteria
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Expected elements for metabolomics queries
        if any(term in query_lower for term in ['metabolomics', 'biomarker', 'metabolite']):
            expected_elements = {
                'analytical_method': ['lc-ms', 'gc-ms', 'nmr', 'mass spectrometry'],
                'statistical_analysis': ['p-value', 'correlation', 'significance'],
                'clinical_relevance': ['disease', 'diagnosis', 'clinical', 'patient']
            }
            
            for element_type, keywords in expected_elements.items():
                if not any(keyword in response_lower for keyword in keywords):
                    validations.append(TestValidationResult(
                        validation_id=f"missing_{element_type}_{int(time.time())}",
                        test_name="response_completeness",
                        validation_type=TestValidationType.RESPONSE_QUALITY,
                        severity=ValidationSeverity.MINOR,
                        passed=False,
                        confidence=0.6,
                        message=f"Missing {element_type.replace('_', ' ')} information",
                        metadata={'element_type': element_type},
                        recommendations=[f"Include information about {element_type.replace('_', ' ')}"]
                    ))
        
        return validations
    
    def _validate_clinical_accuracy(self, 
                                   response: str, 
                                   context: Dict[str, Any]) -> List[TestValidationResult]:
        """Validate clinical accuracy against established medical knowledge."""
        validations = []
        
        # Check for unsubstantiated clinical claims
        dangerous_claim_patterns = [
            r'(cure|treat|prevent)s?\s+(cancer|diabetes|heart\s+disease)',
            r'replace\s+medication',
            r'guaranteed\s+(cure|treatment|results)',
            r'miracle\s+(cure|treatment)',
            r'100%\s+(effective|success|cure)'
        ]
        
        for pattern in dangerous_claim_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                # Check if claim is properly qualified
                surrounding_context = response[max(0, match.start()-100):match.end()+100].lower()
                qualifiers = [
                    'may', 'might', 'could', 'potential', 'preliminary', 
                    'studies suggest', 'research indicates', 'evidence suggests'
                ]
                
                if not any(qualifier in surrounding_context for qualifier in qualifiers):
                    validations.append(TestValidationResult(
                        validation_id=f"clinical_accuracy_{int(time.time())}",
                        test_name="clinical_accuracy_validation",
                        validation_type=TestValidationType.DOMAIN_ACCURACY,
                        severity=ValidationSeverity.CRITICAL,
                        passed=False,
                        confidence=0.95,
                        message=f"Unsubstantiated clinical claim: {match.group()}",
                        evidence=[f"Found pattern: {pattern}", f"Context: {surrounding_context[:200]}"],
                        recommendations=[
                            "Add appropriate qualifiers to clinical claims",
                            "Include supporting evidence or references",
                            "Consider regulatory compliance requirements"
                        ]
                    ))
        
        return validations
    
    def _validate_advanced_terminology(self, response: str) -> List[TestValidationResult]:
        """Validate advanced clinical and scientific terminology usage."""
        validations = []
        
        # Check for proper use of metabolomics terminology
        terminology_checks = {
            'lc-ms': ['LC-MS/MS', 'LC-MS²', 'liquid chromatography'],
            'biomarker': ['biomarkers', 'biological marker', 'disease marker'],
            'metabolite': ['metabolites', 'small molecules', 'biochemicals']
        }
        
        response_lower = response.lower()
        
        for canonical_term, variants in terminology_checks.items():
            found_variants = [variant for variant in variants if variant.lower() in response_lower]
            
            if len(found_variants) > 1:
                # Multiple variants used - check for consistency
                validations.append(TestValidationResult(
                    validation_id=f"terminology_consistency_{int(time.time())}",
                    test_name="terminology_validation",
                    validation_type=TestValidationType.RESPONSE_QUALITY,
                    severity=ValidationSeverity.MINOR,
                    passed=True,  # Not necessarily wrong
                    confidence=0.7,
                    message=f"Multiple terminology variants used for {canonical_term}: {found_variants}",
                    metadata={'canonical_term': canonical_term, 'variants_found': found_variants},
                    recommendations=[
                        "Consider using consistent terminology throughout response",
                        "Define acronyms on first use if appropriate"
                    ]
                ))
        
        return validations
    
    def _map_validation_level_to_severity(self, validation_level: ValidationLevel) -> ValidationSeverity:
        """Map validation level to test validation severity."""
        mapping = {
            ValidationLevel.CRITICAL: ValidationSeverity.CRITICAL,
            ValidationLevel.HIGH: ValidationSeverity.MAJOR,
            ValidationLevel.MEDIUM: ValidationSeverity.MINOR,
            ValidationLevel.LOW: ValidationSeverity.INFO,
            ValidationLevel.INFO: ValidationSeverity.INFO
        }
        return mapping.get(validation_level, ValidationSeverity.WARNING)
    
    def _extract_numerical_values(self, text: str) -> Dict[str, List[float]]:
        """Extract numerical values with their contexts."""
        numerical_data = defaultdict(list)
        
        # Extract various types of numerical values
        patterns = {
            'concentration': re.compile(r'(\d+\.?\d*)\s*(mg/dL|mM|μM|nM|g/L)'),
            'percentage': re.compile(r'(\d+\.?\d*)\s*%'),
            'correlation': re.compile(r'r\s*=\s*([+-]?\d*\.?\d+)'),
            'p_value': re.compile(r'p\s*[<>=]\s*(\d*\.?\d+)'),
            'fold_change': re.compile(r'(\d+\.?\d*)[x×-]?fold')
        }
        
        for value_type, pattern in patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        value = float(match[0])
                    else:
                        value = float(match)
                    numerical_data[value_type].append(value)
                except ValueError:
                    continue
        
        return dict(numerical_data)
    
    def _extract_clinical_claims(self, text: str) -> List[str]:
        """Extract clinical claims from text."""
        claim_patterns = [
            r'(associated with|linked to|correlated with)\s+[^.]{10,50}',
            r'(diagnostic|prognostic|predictive)\s+(marker|biomarker|indicator)',
            r'(treatment|therapy|intervention)\s+for\s+[^.]{10,50}',
            r'(risk factor|protective factor)\s+for\s+[^.]{10,50}'
        ]
        
        claims = []
        for pattern in claim_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            claims.extend([match.group() for match in matches])
        
        return claims
    
    def _validate_metabolite_consistency(self, response_data: List[Dict[str, Any]]) -> List[TestValidationResult]:
        """Validate consistency of metabolite information across responses."""
        validations = []
        
        # Group responses by mentioned metabolites
        metabolite_responses = defaultdict(list)
        for data in response_data:
            for metabolite in data['metabolites']:
                metabolite_responses[metabolite].append(data)
        
        # Check consistency for metabolites mentioned in multiple responses
        for metabolite, responses in metabolite_responses.items():
            if len(responses) > 1:
                # Compare metabolite information across responses
                inconsistencies = self._find_metabolite_inconsistencies(metabolite, responses)
                
                if inconsistencies:
                    validations.append(TestValidationResult(
                        validation_id=f"metabolite_consistency_{metabolite}_{int(time.time())}",
                        test_name="metabolite_consistency",
                        validation_type=TestValidationType.CROSS_TEST_CONSISTENCY,
                        severity=ValidationSeverity.MAJOR,
                        passed=False,
                        confidence=0.8,
                        message=f"Inconsistent information for metabolite {metabolite}",
                        evidence=inconsistencies,
                        metadata={'metabolite': metabolite, 'response_count': len(responses)},
                        recommendations=[
                            f"Verify {metabolite} information for consistency",
                            "Use standardized knowledge base for metabolite data"
                        ]
                    ))
        
        return validations
    
    def _validate_numerical_consistency(self, response_data: List[Dict[str, Any]]) -> List[TestValidationResult]:
        """Validate consistency of numerical values across responses."""
        validations = []
        
        # Compare numerical values across responses
        all_numerical_data = [data['numerical_values'] for data in response_data]
        
        for value_type in ['concentration', 'percentage', 'correlation', 'p_value']:
            values_by_response = []
            for numerical_data in all_numerical_data:
                if value_type in numerical_data:
                    values_by_response.extend(numerical_data[value_type])
            
            if len(values_by_response) > 1:
                # Check for outliers or inconsistencies
                mean_value = statistics.mean(values_by_response)
                std_dev = statistics.stdev(values_by_response) if len(values_by_response) > 1 else 0
                
                outliers = [v for v in values_by_response if abs(v - mean_value) > 2 * std_dev]
                
                if outliers and std_dev > 0.1 * mean_value:  # High variability
                    validations.append(TestValidationResult(
                        validation_id=f"numerical_consistency_{value_type}_{int(time.time())}",
                        test_name="numerical_consistency",
                        validation_type=TestValidationType.STATISTICAL_CONSISTENCY,
                        severity=ValidationSeverity.MINOR,
                        passed=False,
                        confidence=0.7,
                        message=f"High variability in {value_type} values across responses",
                        actual_value=std_dev / mean_value if mean_value != 0 else 0,
                        expected_value=0.1,
                        metadata={
                            'value_type': value_type,
                            'mean': mean_value,
                            'std_dev': std_dev,
                            'outliers': outliers
                        },
                        recommendations=[
                            f"Review {value_type} values for consistency",
                            "Verify data sources and calculation methods"
                        ]
                    ))
        
        return validations
    
    def _validate_clinical_claim_consistency(self, response_data: List[Dict[str, Any]]) -> List[TestValidationResult]:
        """Validate consistency of clinical claims across responses."""
        validations = []
        
        # Extract and compare clinical claims
        all_claims = []
        for data in response_data:
            all_claims.extend(data['clinical_claims'])
        
        if len(all_claims) > 1:
            # Look for contradictory claims
            claim_similarity_matrix = self._calculate_claim_similarity_matrix(all_claims)
            contradictory_pairs = self._find_contradictory_claims(all_claims, claim_similarity_matrix)
            
            if contradictory_pairs:
                validations.append(TestValidationResult(
                    validation_id=f"clinical_claim_consistency_{int(time.time())}",
                    test_name="clinical_claim_consistency",
                    validation_type=TestValidationType.CROSS_TEST_CONSISTENCY,
                    severity=ValidationSeverity.MAJOR,
                    passed=False,
                    confidence=0.8,
                    message="Contradictory clinical claims detected across responses",
                    evidence=[f"Contradictory pair: {pair}" for pair in contradictory_pairs[:3]],
                    metadata={'contradictory_pairs_count': len(contradictory_pairs)},
                    recommendations=[
                        "Review clinical claims for consistency",
                        "Establish standardized clinical knowledge base",
                        "Implement claim verification process"
                    ]
                ))
        
        return validations
    
    def _find_metabolite_inconsistencies(self, metabolite: str, responses: List[Dict[str, Any]]) -> List[str]:
        """Find inconsistencies in metabolite information."""
        inconsistencies = []
        
        # Extract metabolite-specific information from each response
        metabolite_info = []
        for response_data in responses:
            response_text = response_data['response']
            
            # Look for molecular formula mentions
            formula_pattern = rf'{metabolite}.*?([A-Z]\d*[A-Z]*\d*)'
            formula_matches = re.findall(formula_pattern, response_text, re.IGNORECASE)
            
            # Look for molecular weight mentions
            weight_pattern = rf'{metabolite}.*?(\d+\.?\d*)\s*(da|dalton)'
            weight_matches = re.findall(weight_pattern, response_text, re.IGNORECASE)
            
            metabolite_info.append({
                'response_index': response_data['index'],
                'formulas': formula_matches,
                'weights': weight_matches
            })
        
        # Compare information for inconsistencies
        if len(set(tuple(info['formulas']) for info in metabolite_info if info['formulas'])) > 1:
            inconsistencies.append(f"Inconsistent molecular formulas for {metabolite}")
        
        if len(set(tuple(info['weights']) for info in metabolite_info if info['weights'])) > 1:
            inconsistencies.append(f"Inconsistent molecular weights for {metabolite}")
        
        return inconsistencies
    
    def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """Calculate similarity between two responses."""
        # Simple word overlap similarity
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_claim_similarity_matrix(self, claims: List[str]) -> np.ndarray:
        """Calculate similarity matrix for clinical claims."""
        n = len(claims)
        similarity_matrix = np.ones((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._calculate_response_similarity(claims[i], claims[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def _find_contradictory_claims(self, claims: List[str], similarity_matrix: np.ndarray) -> List[Tuple[str, str]]:
        """Find contradictory claims based on similarity and logical analysis."""
        contradictory_pairs = []
        n = len(claims)
        
        # Look for claims with very low similarity that might be contradictory
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] < 0.2:  # Very low similarity
                    # Check if they're actually contradictory vs just different topics
                    if self._are_claims_contradictory(claims[i], claims[j]):
                        contradictory_pairs.append((claims[i], claims[j]))
        
        return contradictory_pairs
    
    def _are_claims_contradictory(self, claim1: str, claim2: str) -> bool:
        """Determine if two claims are contradictory."""
        # Simple heuristic: look for opposite relationships or negations
        contradictory_patterns = [
            ('associated with', 'not associated with'),
            ('increases', 'decreases'),
            ('elevated', 'decreased'),
            ('high', 'low'),
            ('positive', 'negative')
        ]
        
        claim1_lower = claim1.lower()
        claim2_lower = claim2.lower()
        
        for positive_pattern, negative_pattern in contradictory_patterns:
            if positive_pattern in claim1_lower and negative_pattern in claim2_lower:
                return True
            if negative_pattern in claim1_lower and positive_pattern in claim2_lower:
                return True
        
        return False


# =====================================================================
# TEST RESULT VALIDATOR
# =====================================================================

class TestResultValidator:
    """
    Comprehensive test result validator for result pattern validation,
    statistical result validation, and cross-test consistency checking.
    """
    
    def __init__(self, 
                 performance_assertion_helper: Optional[PerformanceAssertionHelper] = None):
        self.performance_helper = performance_assertion_helper or PerformanceAssertionHelper()
        self.validation_rules: Dict[str, List[ValidationAssertion]] = {}
        self.test_result_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.statistical_thresholds = self._load_statistical_thresholds()
        self.logger = logging.getLogger(__name__)
    
    def _load_statistical_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load statistical validation thresholds."""
        return {
            'correlation': {
                'min_valid': -1.0,
                'max_valid': 1.0,
                'strong_threshold': 0.7,
                'moderate_threshold': 0.5
            },
            'p_value': {
                'min_valid': 0.0,
                'max_valid': 1.0,
                'significance_threshold': 0.05,
                'high_significance_threshold': 0.01
            },
            'auc': {
                'min_valid': 0.0,
                'max_valid': 1.0,
                'excellent_threshold': 0.9,
                'good_threshold': 0.8,
                'fair_threshold': 0.7
            },
            'sensitivity_specificity': {
                'min_valid': 0.0,
                'max_valid': 1.0,
                'clinical_threshold': 0.8,
                'excellent_threshold': 0.95
            }
        }
    
    def validate_result_patterns(self, 
                                test_results: Dict[str, Any],
                                expected_patterns: Dict[str, Any]) -> List[TestValidationResult]:
        """Validate test results against expected patterns."""
        validations = []
        
        for pattern_name, pattern_spec in expected_patterns.items():
            validation_result = self._validate_single_pattern(
                test_results, pattern_name, pattern_spec
            )
            validations.append(validation_result)
        
        return validations
    
    def validate_statistical_results(self, 
                                   statistical_data: Dict[str, Any],
                                   validation_criteria: Optional[Dict[str, Any]] = None) -> List[TestValidationResult]:
        """Validate statistical results for correctness and consistency."""
        validations = []
        
        # Validate correlation coefficients
        if 'correlation' in statistical_data:
            correlation_validation = self._validate_correlation_coefficients(
                statistical_data['correlation']
            )
            validations.extend(correlation_validation)
        
        # Validate p-values
        if 'p_values' in statistical_data:
            p_value_validation = self._validate_p_values(statistical_data['p_values'])
            validations.extend(p_value_validation)
        
        # Validate AUC values
        if 'auc_values' in statistical_data:
            auc_validation = self._validate_auc_values(statistical_data['auc_values'])
            validations.extend(auc_validation)
        
        # Validate sensitivity/specificity
        if 'sensitivity' in statistical_data or 'specificity' in statistical_data:
            diagnostic_validation = self._validate_diagnostic_metrics(statistical_data)
            validations.extend(diagnostic_validation)
        
        # Cross-validate statistical consistency
        consistency_validation = self._validate_statistical_consistency(statistical_data)
        validations.extend(consistency_validation)
        
        return validations
    
    def validate_cross_test_consistency(self, 
                                      test_results: List[Dict[str, Any]],
                                      consistency_rules: Optional[Dict[str, Any]] = None) -> CrossTestValidationResult:
        """Validate consistency across multiple test results."""
        
        if len(test_results) < 2:
            return CrossTestValidationResult(
                validation_suite="cross_test_consistency",
                tests_compared=[],
                consistency_score=1.0,
                inconsistencies_found=[],
                recommendations=["Need at least 2 test results for consistency validation"]
            )
        
        test_names = [result.get('test_name', f'test_{i}') for i, result in enumerate(test_results)]
        inconsistencies = []
        
        # Check performance consistency
        performance_inconsistencies = self._check_performance_consistency(test_results)
        inconsistencies.extend(performance_inconsistencies)
        
        # Check result pattern consistency
        pattern_inconsistencies = self._check_pattern_consistency(test_results)
        inconsistencies.extend(pattern_inconsistencies)
        
        # Check statistical consistency
        statistical_inconsistencies = self._check_statistical_consistency(test_results)
        inconsistencies.extend(statistical_inconsistencies)
        
        # Calculate overall consistency score
        total_possible_checks = len(test_results) * (len(test_results) - 1) // 2
        consistency_score = max(0.0, 1.0 - len(inconsistencies) / max(1, total_possible_checks))
        
        # Generate recommendations
        recommendations = self._generate_consistency_recommendations(inconsistencies)
        
        return CrossTestValidationResult(
            validation_suite="cross_test_consistency",
            tests_compared=test_names,
            consistency_score=consistency_score,
            inconsistencies_found=inconsistencies,
            recommendations=recommendations
        )
    
    def add_custom_validation_rule(self, 
                                  rule_name: str, 
                                  validation_function: Callable,
                                  expected_result: bool = True,
                                  severity: ValidationSeverity = ValidationSeverity.MAJOR,
                                  description: str = "") -> None:
        """Add custom validation rule."""
        
        if rule_name not in self.validation_rules:
            self.validation_rules[rule_name] = []
        
        assertion = ValidationAssertion(
            assertion_name=rule_name,
            validation_function=validation_function,
            expected_result=expected_result,
            severity=severity,
            description=description or f"Custom validation rule: {rule_name}"
        )
        
        self.validation_rules[rule_name].append(assertion)
        self.logger.info(f"Added custom validation rule: {rule_name}")
    
    def apply_custom_validation_rules(self, 
                                    test_data: Dict[str, Any],
                                    rule_category: Optional[str] = None) -> List[TestValidationResult]:
        """Apply custom validation rules to test data."""
        validations = []
        
        rules_to_apply = self.validation_rules
        if rule_category and rule_category in self.validation_rules:
            rules_to_apply = {rule_category: self.validation_rules[rule_category]}
        
        for rule_name, assertions in rules_to_apply.items():
            for assertion in assertions:
                try:
                    result = assertion.validation_function(test_data, **assertion.parameters)
                    
                    passed = (result == assertion.expected_result)
                    
                    validation = TestValidationResult(
                        validation_id=f"custom_{rule_name}_{int(time.time())}",
                        test_name="custom_validation",
                        validation_type=TestValidationType.PATTERN_VALIDATION,
                        severity=assertion.severity,
                        passed=passed,
                        confidence=0.9,
                        message=f"Custom validation {rule_name}: {'PASSED' if passed else 'FAILED'}",
                        actual_value=result,
                        expected_value=assertion.expected_result,
                        metadata={
                            'rule_name': rule_name,
                            'rule_description': assertion.description
                        }
                    )
                    
                    validations.append(validation)
                    
                except Exception as e:
                    validation = TestValidationResult(
                        validation_id=f"custom_{rule_name}_error_{int(time.time())}",
                        test_name="custom_validation",
                        validation_type=TestValidationType.PATTERN_VALIDATION,
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        confidence=0.5,
                        message=f"Custom validation {rule_name} failed with error: {str(e)}",
                        metadata={
                            'rule_name': rule_name,
                            'error': str(e)
                        },
                        recommendations=[f"Fix custom validation rule: {rule_name}"]
                    )
                    validations.append(validation)
        
        return validations
    
    def _validate_single_pattern(self, 
                               test_results: Dict[str, Any], 
                               pattern_name: str, 
                               pattern_spec: Dict[str, Any]) -> TestValidationResult:
        """Validate a single result pattern."""
        
        try:
            # Extract the value to validate
            if 'path' in pattern_spec:
                actual_value = self._extract_value_by_path(test_results, pattern_spec['path'])
            else:
                actual_value = test_results.get(pattern_name)
            
            expected_value = pattern_spec.get('expected_value')
            validation_type = pattern_spec.get('validation_type', 'equals')
            tolerance = pattern_spec.get('tolerance', 0.0)
            
            passed = self._validate_value(actual_value, expected_value, validation_type, tolerance)
            
            return TestValidationResult(
                validation_id=f"pattern_{pattern_name}_{int(time.time())}",
                test_name="result_pattern_validation",
                validation_type=TestValidationType.PATTERN_VALIDATION,
                severity=ValidationSeverity.MAJOR,
                passed=passed,
                confidence=0.9,
                message=f"Pattern validation {pattern_name}: {'PASSED' if passed else 'FAILED'}",
                actual_value=actual_value,
                expected_value=expected_value,
                tolerance=tolerance,
                metadata={
                    'pattern_name': pattern_name,
                    'validation_type': validation_type
                }
            )
            
        except Exception as e:
            return TestValidationResult(
                validation_id=f"pattern_{pattern_name}_error_{int(time.time())}",
                test_name="result_pattern_validation",
                validation_type=TestValidationType.PATTERN_VALIDATION,
                severity=ValidationSeverity.WARNING,
                passed=False,
                confidence=0.5,
                message=f"Pattern validation {pattern_name} failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    def _validate_correlation_coefficients(self, correlations: Dict[str, float]) -> List[TestValidationResult]:
        """Validate correlation coefficient values."""
        validations = []
        thresholds = self.statistical_thresholds['correlation']
        
        for corr_name, corr_value in correlations.items():
            # Check valid range
            if not (thresholds['min_valid'] <= corr_value <= thresholds['max_valid']):
                validations.append(TestValidationResult(
                    validation_id=f"correlation_range_{corr_name}_{int(time.time())}",
                    test_name="correlation_validation",
                    validation_type=TestValidationType.STATISTICAL_CONSISTENCY,
                    severity=ValidationSeverity.CRITICAL,
                    passed=False,
                    confidence=1.0,
                    message=f"Correlation coefficient {corr_name} outside valid range: {corr_value}",
                    actual_value=corr_value,
                    expected_value=f"Range: {thresholds['min_valid']} to {thresholds['max_valid']}",
                    recommendations=[f"Correct correlation coefficient {corr_name} to be within [-1, 1]"]
                ))
            
            # Assess correlation strength
            elif abs(corr_value) >= thresholds['strong_threshold']:
                validations.append(TestValidationResult(
                    validation_id=f"correlation_strong_{corr_name}_{int(time.time())}",
                    test_name="correlation_validation",
                    validation_type=TestValidationType.STATISTICAL_CONSISTENCY,
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    confidence=0.9,
                    message=f"Strong correlation detected for {corr_name}: {corr_value}",
                    actual_value=abs(corr_value),
                    metadata={'correlation_strength': 'strong'}
                ))
        
        return validations
    
    def _validate_p_values(self, p_values: Dict[str, float]) -> List[TestValidationResult]:
        """Validate p-value correctness."""
        validations = []
        thresholds = self.statistical_thresholds['p_value']
        
        for p_name, p_value in p_values.items():
            # Check valid range
            if not (thresholds['min_valid'] <= p_value <= thresholds['max_valid']):
                validations.append(TestValidationResult(
                    validation_id=f"p_value_range_{p_name}_{int(time.time())}",
                    test_name="p_value_validation",
                    validation_type=TestValidationType.STATISTICAL_CONSISTENCY,
                    severity=ValidationSeverity.CRITICAL,
                    passed=False,
                    confidence=1.0,
                    message=f"P-value {p_name} outside valid range: {p_value}",
                    actual_value=p_value,
                    expected_value=f"Range: {thresholds['min_valid']} to {thresholds['max_valid']}",
                    recommendations=[f"Correct p-value {p_name} to be within [0, 1]"]
                ))
            
            # Flag exactly zero p-values as suspicious
            elif p_value == 0.0:
                validations.append(TestValidationResult(
                    validation_id=f"p_value_zero_{p_name}_{int(time.time())}",
                    test_name="p_value_validation",
                    validation_type=TestValidationType.STATISTICAL_CONSISTENCY,
                    severity=ValidationSeverity.MINOR,
                    passed=False,
                    confidence=0.8,
                    message=f"P-value of exactly 0.0 is statistically implausible for {p_name}",
                    actual_value=p_value,
                    recommendations=[f"Use p < 0.001 instead of p = 0.0 for {p_name}"]
                ))
            
            # Assess statistical significance
            elif p_value <= thresholds['high_significance_threshold']:
                validations.append(TestValidationResult(
                    validation_id=f"p_value_significant_{p_name}_{int(time.time())}",
                    test_name="p_value_validation",
                    validation_type=TestValidationType.STATISTICAL_CONSISTENCY,
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    confidence=0.95,
                    message=f"Highly significant p-value for {p_name}: {p_value}",
                    actual_value=p_value,
                    metadata={'significance_level': 'highly_significant'}
                ))
        
        return validations
    
    def _validate_auc_values(self, auc_values: Dict[str, float]) -> List[TestValidationResult]:
        """Validate AUC (Area Under Curve) values."""
        validations = []
        thresholds = self.statistical_thresholds['auc']
        
        for auc_name, auc_value in auc_values.items():
            # Check valid range
            if not (thresholds['min_valid'] <= auc_value <= thresholds['max_valid']):
                validations.append(TestValidationResult(
                    validation_id=f"auc_range_{auc_name}_{int(time.time())}",
                    test_name="auc_validation",
                    validation_type=TestValidationType.STATISTICAL_CONSISTENCY,
                    severity=ValidationSeverity.CRITICAL,
                    passed=False,
                    confidence=1.0,
                    message=f"AUC value {auc_name} outside valid range: {auc_value}",
                    actual_value=auc_value,
                    expected_value=f"Range: {thresholds['min_valid']} to {thresholds['max_valid']}",
                    recommendations=[f"Correct AUC value {auc_name} to be within [0, 1]"]
                ))
            
            # Assess AUC performance level
            else:
                if auc_value >= thresholds['excellent_threshold']:
                    performance_level = 'excellent'
                    severity = ValidationSeverity.INFO
                elif auc_value >= thresholds['good_threshold']:
                    performance_level = 'good'
                    severity = ValidationSeverity.INFO
                elif auc_value >= thresholds['fair_threshold']:
                    performance_level = 'fair'
                    severity = ValidationSeverity.MINOR
                else:
                    performance_level = 'poor'
                    severity = ValidationSeverity.MAJOR
                
                validations.append(TestValidationResult(
                    validation_id=f"auc_performance_{auc_name}_{int(time.time())}",
                    test_name="auc_validation",
                    validation_type=TestValidationType.STATISTICAL_CONSISTENCY,
                    severity=severity,
                    passed=performance_level in ['excellent', 'good', 'fair'],
                    confidence=0.9,
                    message=f"AUC performance for {auc_name}: {performance_level} ({auc_value:.3f})",
                    actual_value=auc_value,
                    metadata={'performance_level': performance_level}
                ))
        
        return validations
    
    def _validate_diagnostic_metrics(self, statistical_data: Dict[str, Any]) -> List[TestValidationResult]:
        """Validate diagnostic metrics (sensitivity, specificity, etc.)."""
        validations = []
        thresholds = self.statistical_thresholds['sensitivity_specificity']
        
        diagnostic_metrics = ['sensitivity', 'specificity', 'precision', 'recall', 'f1_score']
        
        for metric_name in diagnostic_metrics:
            if metric_name in statistical_data:
                metric_value = statistical_data[metric_name]
                
                # Handle both single values and dictionaries
                if isinstance(metric_value, dict):
                    for sub_name, sub_value in metric_value.items():
                        validation = self._validate_single_diagnostic_metric(
                            f"{metric_name}_{sub_name}", sub_value, thresholds
                        )
                        validations.append(validation)
                else:
                    validation = self._validate_single_diagnostic_metric(
                        metric_name, metric_value, thresholds
                    )
                    validations.append(validation)
        
        return validations
    
    def _validate_single_diagnostic_metric(self, 
                                         metric_name: str, 
                                         metric_value: float, 
                                         thresholds: Dict[str, float]) -> TestValidationResult:
        """Validate a single diagnostic metric."""
        
        # Check valid range
        if not (thresholds['min_valid'] <= metric_value <= thresholds['max_valid']):
            return TestValidationResult(
                validation_id=f"diagnostic_range_{metric_name}_{int(time.time())}",
                test_name="diagnostic_validation",
                validation_type=TestValidationType.STATISTICAL_CONSISTENCY,
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                confidence=1.0,
                message=f"Diagnostic metric {metric_name} outside valid range: {metric_value}",
                actual_value=metric_value,
                expected_value=f"Range: {thresholds['min_valid']} to {thresholds['max_valid']}",
                recommendations=[f"Correct {metric_name} to be within [0, 1]"]
            )
        
        # Assess clinical utility
        if metric_value >= thresholds['excellent_threshold']:
            clinical_utility = 'excellent'
            severity = ValidationSeverity.INFO
        elif metric_value >= thresholds['clinical_threshold']:
            clinical_utility = 'clinically_useful'
            severity = ValidationSeverity.INFO
        else:
            clinical_utility = 'below_clinical_threshold'
            severity = ValidationSeverity.MINOR
        
        return TestValidationResult(
            validation_id=f"diagnostic_utility_{metric_name}_{int(time.time())}",
            test_name="diagnostic_validation",
            validation_type=TestValidationType.STATISTICAL_CONSISTENCY,
            severity=severity,
            passed=clinical_utility in ['excellent', 'clinically_useful'],
            confidence=0.8,
            message=f"Diagnostic metric {metric_name}: {clinical_utility} ({metric_value:.3f})",
            actual_value=metric_value,
            metadata={'clinical_utility': clinical_utility}
        )
    
    def _validate_statistical_consistency(self, statistical_data: Dict[str, Any]) -> List[TestValidationResult]:
        """Validate overall statistical consistency."""
        validations = []
        
        # Check for logical consistency between related metrics
        if 'sensitivity' in statistical_data and 'specificity' in statistical_data:
            sensitivity = statistical_data['sensitivity']
            specificity = statistical_data['specificity']
            
            # Check if both are unusually high (might indicate overfitting)
            if isinstance(sensitivity, (int, float)) and isinstance(specificity, (int, float)):
                if sensitivity > 0.98 and specificity > 0.98:
                    validations.append(TestValidationResult(
                        validation_id=f"overfitting_check_{int(time.time())}",
                        test_name="statistical_consistency",
                        validation_type=TestValidationType.STATISTICAL_CONSISTENCY,
                        severity=ValidationSeverity.MINOR,
                        passed=False,
                        confidence=0.7,
                        message=f"Unusually high sensitivity ({sensitivity}) and specificity ({specificity}) - possible overfitting",
                        metadata={
                            'sensitivity': sensitivity,
                            'specificity': specificity,
                            'potential_issue': 'overfitting'
                        },
                        recommendations=[
                            "Verify results with independent validation set",
                            "Check for data leakage or overfitting"
                        ]
                    ))
        
        return validations
    
    def _check_performance_consistency(self, test_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check performance consistency across tests."""
        inconsistencies = []
        
        # Extract performance metrics from each test
        performance_metrics = []
        for result in test_results:
            if 'performance_metrics' in result:
                performance_metrics.append(result['performance_metrics'])
        
        if len(performance_metrics) > 1:
            # Check for significant performance variations
            for metric_name in ['response_time_ms', 'memory_usage_mb', 'throughput_ops_per_sec']:
                values = []
                for metrics in performance_metrics:
                    if metric_name in metrics:
                        values.append(metrics[metric_name])
                
                if len(values) > 1:
                    mean_value = statistics.mean(values)
                    std_dev = statistics.stdev(values)
                    
                    # Check for high variability
                    coefficient_of_variation = std_dev / mean_value if mean_value != 0 else 0
                    
                    if coefficient_of_variation > 0.3:  # High variability
                        inconsistencies.append({
                            'type': 'performance_inconsistency',
                            'metric': metric_name,
                            'coefficient_of_variation': coefficient_of_variation,
                            'values': values,
                            'severity': 'major'
                        })
        
        return inconsistencies
    
    def _check_pattern_consistency(self, test_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check pattern consistency across test results."""
        inconsistencies = []
        
        # Compare result patterns across tests
        patterns = []
        for result in test_results:
            if 'result_patterns' in result:
                patterns.append(result['result_patterns'])
        
        if len(patterns) > 1:
            # Find common pattern keys
            common_keys = set(patterns[0].keys())
            for pattern in patterns[1:]:
                common_keys = common_keys.intersection(set(pattern.keys()))
            
            # Check consistency for common keys
            for key in common_keys:
                values = [pattern[key] for pattern in patterns]
                unique_values = set(str(v) for v in values)  # Convert to string for comparison
                
                if len(unique_values) > 1:
                    inconsistencies.append({
                        'type': 'pattern_inconsistency',
                        'pattern_key': key,
                        'values': values,
                        'unique_values': list(unique_values),
                        'severity': 'minor'
                    })
        
        return inconsistencies
    
    def _check_statistical_consistency(self, test_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check statistical consistency across test results."""
        inconsistencies = []
        
        # Extract statistical data from each test
        statistical_datasets = []
        for result in test_results:
            if 'statistical_data' in result:
                statistical_datasets.append(result['statistical_data'])
        
        if len(statistical_datasets) > 1:
            # Check for consistent statistical relationships
            for stat_type in ['correlation', 'p_values', 'auc_values']:
                all_values = []
                for dataset in statistical_datasets:
                    if stat_type in dataset:
                        if isinstance(dataset[stat_type], dict):
                            all_values.extend(dataset[stat_type].values())
                        else:
                            all_values.append(dataset[stat_type])
                
                if len(all_values) > 1:
                    # Check for outliers or inconsistencies
                    mean_value = statistics.mean(all_values)
                    std_dev = statistics.stdev(all_values) if len(all_values) > 1 else 0
                    
                    outliers = [v for v in all_values if abs(v - mean_value) > 2 * std_dev]
                    
                    if outliers and std_dev > 0.1:
                        inconsistencies.append({
                            'type': 'statistical_inconsistency',
                            'statistic_type': stat_type,
                            'outliers': outliers,
                            'mean': mean_value,
                            'std_dev': std_dev,
                            'severity': 'minor'
                        })
        
        return inconsistencies
    
    def _generate_consistency_recommendations(self, inconsistencies: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on consistency check results."""
        recommendations = []
        
        # Group inconsistencies by type
        inconsistency_types = defaultdict(list)
        for inconsistency in inconsistencies:
            inconsistency_types[inconsistency['type']].append(inconsistency)
        
        # Generate type-specific recommendations
        if 'performance_inconsistency' in inconsistency_types:
            recommendations.append("Review test environments for performance consistency")
            recommendations.append("Consider using standardized test configurations")
        
        if 'pattern_inconsistency' in inconsistency_types:
            recommendations.append("Establish consistent result pattern expectations")
            recommendations.append("Review test implementation for pattern variations")
        
        if 'statistical_inconsistency' in inconsistency_types:
            recommendations.append("Investigate statistical outliers across test runs")
            recommendations.append("Consider using statistical confidence intervals")
        
        # General recommendations
        if len(inconsistencies) > 5:
            recommendations.append("Implement stricter consistency validation rules")
            recommendations.append("Consider automated consistency monitoring")
        
        return recommendations[:10]  # Limit to top 10
    
    def _extract_value_by_path(self, data: Dict[str, Any], path: str) -> Any:
        """Extract value from nested dictionary using dot notation path."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _validate_value(self, 
                       actual: Any, 
                       expected: Any, 
                       validation_type: str, 
                       tolerance: float) -> bool:
        """Validate a value against expected criteria."""
        
        if validation_type == 'equals':
            if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                return abs(actual - expected) <= tolerance
            else:
                return actual == expected
        
        elif validation_type == 'greater_than':
            return isinstance(actual, (int, float)) and actual > expected
        
        elif validation_type == 'less_than':
            return isinstance(actual, (int, float)) and actual < expected
        
        elif validation_type == 'in_range':
            if isinstance(expected, (list, tuple)) and len(expected) == 2:
                return expected[0] <= actual <= expected[1]
            return False
        
        elif validation_type == 'contains':
            return expected in str(actual)
        
        elif validation_type == 'regex':
            return bool(re.match(str(expected), str(actual)))
        
        else:
            return actual == expected


# =====================================================================
# CLINICAL METABOLOMICS VALIDATOR
# =====================================================================

class ClinicalMetabolomicsValidator:
    """
    Domain-specific validator for clinical metabolomics content and results.
    """
    
    def __init__(self, data_generator: Optional[ClinicalMetabolomicsDataGenerator] = None):
        self.data_generator = data_generator or ClinicalMetabolomicsDataGenerator()
        self.metabolite_knowledge = self.data_generator.METABOLITE_DATABASE
        self.disease_panels = self.data_generator.DISEASE_PANELS
        self.analytical_platforms = self.data_generator.ANALYTICAL_PLATFORMS
        self.logger = logging.getLogger(__name__)
    
    def validate_metabolomics_query_response(self, 
                                           query: str, 
                                           response: str,
                                           expected_metabolites: Optional[List[str]] = None) -> List[TestValidationResult]:
        """Validate response to metabolomics query for domain accuracy."""
        validations = []
        
        # Extract mentioned metabolites from response
        mentioned_metabolites = self._extract_mentioned_metabolites(response)
        
        # Validate metabolite coverage
        if expected_metabolites:
            coverage_validation = self._validate_metabolite_coverage(
                mentioned_metabolites, expected_metabolites
            )
            validations.extend(coverage_validation)
        
        # Validate metabolite information accuracy
        for metabolite in mentioned_metabolites:
            metabolite_validation = self._validate_metabolite_accuracy(metabolite, response)
            validations.extend(metabolite_validation)
        
        # Validate analytical method mentions
        analytical_validation = self._validate_analytical_methods(response)
        validations.extend(analytical_validation)
        
        # Validate clinical context
        clinical_validation = self._validate_clinical_metabolomics_context(response)
        validations.extend(clinical_validation)
        
        return validations
    
    def validate_biomarker_claims(self, 
                                response: str,
                                disease_context: Optional[str] = None) -> List[TestValidationResult]:
        """Validate biomarker claims for clinical accuracy."""
        validations = []
        
        # Extract biomarker claims
        biomarker_claims = self._extract_biomarker_claims(response)
        
        for claim in biomarker_claims:
            # Validate against known disease-biomarker associations
            if disease_context and disease_context in self.disease_panels:
                disease_panel = self.disease_panels[disease_context]
                
                claim_validation = self._validate_biomarker_claim_accuracy(
                    claim, disease_panel
                )
                validations.append(claim_validation)
        
        return validations
    
    def validate_pathway_information(self, response: str) -> List[TestValidationResult]:
        """Validate metabolic pathway information."""
        validations = []
        
        # Extract pathway mentions
        pathway_mentions = self._extract_pathway_mentions(response)
        
        for pathway in pathway_mentions:
            # Validate pathway-metabolite associations
            pathway_validation = self._validate_pathway_metabolite_associations(
                pathway, response
            )
            validations.extend(pathway_validation)
        
        return validations
    
    def validate_concentration_data(self, 
                                  concentration_data: Dict[str, Any],
                                  expected_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> List[TestValidationResult]:
        """Validate metabolite concentration data."""
        validations = []
        
        for metabolite, data in concentration_data.items():
            if isinstance(data, dict) and 'concentrations' in data:
                concentrations = data['concentrations']
                
                # Validate concentration range
                range_validation = self._validate_concentration_range(
                    metabolite, concentrations, expected_ranges
                )
                validations.extend(range_validation)
                
                # Validate statistical properties
                stats_validation = self._validate_concentration_statistics(
                    metabolite, concentrations
                )
                validations.extend(stats_validation)
        
        return validations
    
    def _extract_mentioned_metabolites(self, text: str) -> List[str]:
        """Extract metabolites mentioned in text."""
        mentioned = []
        text_lower = text.lower()
        
        for metabolite_name in self.metabolite_knowledge.keys():
            if metabolite_name in text_lower:
                mentioned.append(metabolite_name)
            
            # Check synonyms
            metabolite_data = self.metabolite_knowledge[metabolite_name]
            
            # Handle both dict and MetaboliteData object
            synonyms = []
            if hasattr(metabolite_data, 'synonyms'):
                synonyms = getattr(metabolite_data, 'synonyms', []) or []
            elif isinstance(metabolite_data, dict):
                synonyms = metabolite_data.get('synonyms', [])
            
            for synonym in synonyms:
                if synonym.lower() in text_lower and metabolite_name not in mentioned:
                    mentioned.append(metabolite_name)
                    break
        
        return mentioned
    
    def _validate_metabolite_coverage(self, 
                                    mentioned_metabolites: List[str], 
                                    expected_metabolites: List[str]) -> List[TestValidationResult]:
        """Validate coverage of expected metabolites."""
        validations = []
        
        missing_metabolites = [m for m in expected_metabolites if m not in mentioned_metabolites]
        coverage_rate = 1.0 - len(missing_metabolites) / len(expected_metabolites)
        
        if coverage_rate < 0.8:  # Less than 80% coverage
            validations.append(TestValidationResult(
                validation_id=f"metabolite_coverage_{int(time.time())}",
                test_name="metabolite_coverage",
                validation_type=TestValidationType.DOMAIN_ACCURACY,
                severity=ValidationSeverity.MAJOR,
                passed=False,
                confidence=0.9,
                message=f"Low metabolite coverage: {coverage_rate:.1%}",
                actual_value=coverage_rate,
                expected_value=0.8,
                evidence=[f"Missing metabolites: {missing_metabolites}"],
                recommendations=[f"Include information about: {', '.join(missing_metabolites[:3])}"]
            ))
        else:
            validations.append(TestValidationResult(
                validation_id=f"metabolite_coverage_good_{int(time.time())}",
                test_name="metabolite_coverage",
                validation_type=TestValidationType.DOMAIN_ACCURACY,
                severity=ValidationSeverity.INFO,
                passed=True,
                confidence=0.9,
                message=f"Good metabolite coverage: {coverage_rate:.1%}",
                actual_value=coverage_rate,
                metadata={'mentioned_metabolites': mentioned_metabolites}
            ))
        
        return validations
    
    def _validate_metabolite_accuracy(self, metabolite: str, response: str) -> List[TestValidationResult]:
        """Validate accuracy of metabolite information in response."""
        validations = []
        
        if metabolite not in self.metabolite_knowledge:
            return validations
        
        metabolite_data = self.metabolite_knowledge[metabolite]
        
        # Get molecular formula - handle both dict and MetaboliteData object
        molecular_formula = None
        if hasattr(metabolite_data, 'molecular_formula'):
            molecular_formula = getattr(metabolite_data, 'molecular_formula', None)
        elif isinstance(metabolite_data, dict):
            molecular_formula = metabolite_data.get('molecular_formula')
        
        # Check molecular formula if mentioned and available
        if molecular_formula:
            formula_pattern = rf'{metabolite}.*?({molecular_formula})'
            if re.search(formula_pattern, response, re.IGNORECASE):
                validations.append(TestValidationResult(
                    validation_id=f"formula_accuracy_{metabolite}_{int(time.time())}",
                    test_name="metabolite_formula_accuracy",
                    validation_type=TestValidationType.DOMAIN_ACCURACY,
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    confidence=0.95,
                    message=f"Correct molecular formula for {metabolite}",
                    metadata={'metabolite': metabolite, 'formula': molecular_formula}
                ))
        
        # Get pathways - handle both dict and MetaboliteData object
        pathways = []
        if hasattr(metabolite_data, 'pathways'):
            pathways = getattr(metabolite_data, 'pathways', []) or []
        elif isinstance(metabolite_data, dict):
            pathways = metabolite_data.get('pathways', [])
        
        # Check pathways if mentioned
        for pathway in pathways:
            pathway_pattern = rf'{metabolite}.*{pathway.replace("_", "\\s+")}'
            if re.search(pathway_pattern, response, re.IGNORECASE):
                validations.append(TestValidationResult(
                    validation_id=f"pathway_accuracy_{metabolite}_{pathway}_{int(time.time())}",
                    test_name="metabolite_pathway_accuracy",
                    validation_type=TestValidationType.DOMAIN_ACCURACY,
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    confidence=0.8,
                    message=f"Correct pathway association: {metabolite} - {pathway}",
                    metadata={'metabolite': metabolite, 'pathway': pathway}
                ))
        
        return validations
    
    def _validate_analytical_methods(self, response: str) -> List[TestValidationResult]:
        """Validate mentions of analytical methods."""
        validations = []
        response_lower = response.lower()
        
        mentioned_methods = []
        for platform_name in self.analytical_platforms.keys():
            if platform_name.lower().replace('-', ' ') in response_lower:
                mentioned_methods.append(platform_name)
        
        if mentioned_methods:
            for method in mentioned_methods:
                platform_data = self.analytical_platforms[method]
                
                validations.append(TestValidationResult(
                    validation_id=f"analytical_method_{method}_{int(time.time())}",
                    test_name="analytical_method_validation",
                    validation_type=TestValidationType.DOMAIN_ACCURACY,
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    confidence=0.8,
                    message=f"Valid analytical method mentioned: {method}",
                    metadata={
                        'method': method,
                        'full_name': platform_data['full_name'],
                        'advantages': platform_data['advantages']
                    }
                ))
        
        return validations
    
    def _validate_clinical_metabolomics_context(self, response: str) -> List[TestValidationResult]:
        """Validate clinical context in metabolomics response."""
        validations = []
        response_lower = response.lower()
        
        # Check for clinical relevance indicators
        clinical_indicators = [
            'diagnosis', 'biomarker', 'patient', 'clinical', 'disease',
            'therapeutic', 'prognosis', 'treatment', 'screening'
        ]
        
        mentioned_indicators = [ind for ind in clinical_indicators if ind in response_lower]
        
        if mentioned_indicators:
            clinical_relevance_score = len(mentioned_indicators) / len(clinical_indicators)
            
            validations.append(TestValidationResult(
                validation_id=f"clinical_context_{int(time.time())}",
                test_name="clinical_context_validation",
                validation_type=TestValidationType.DOMAIN_ACCURACY,
                severity=ValidationSeverity.INFO,
                passed=clinical_relevance_score >= 0.2,
                confidence=0.7,
                message=f"Clinical context score: {clinical_relevance_score:.1%}",
                actual_value=clinical_relevance_score,
                metadata={'clinical_indicators': mentioned_indicators}
            ))
        
        return validations
    
    def _extract_biomarker_claims(self, response: str) -> List[str]:
        """Extract biomarker claims from response."""
        biomarker_patterns = [
            r'(\w+)\s+(?:is|are|serve[s]?\s+as)\s+(?:a\s+)?biomarker[s]?\s+for\s+(\w+)',
            r'biomarker[s]?\s+for\s+(\w+)\s+include[s]?\s+([^.]+)',
            r'(\w+)\s+level[s]?\s+(?:are\s+)?(?:elevated|increased|decreased)\s+in\s+(\w+)'
        ]
        
        claims = []
        for pattern in biomarker_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            claims.extend([match.group() for match in matches])
        
        return claims
    
    def _validate_biomarker_claim_accuracy(self, 
                                         claim: str, 
                                         disease_panel: Dict[str, Any]) -> TestValidationResult:
        """Validate accuracy of a biomarker claim."""
        
        claim_lower = claim.lower()
        
        # Check if claimed biomarkers are in the disease panel
        primary_markers = disease_panel.get('primary_markers', [])
        secondary_markers = disease_panel.get('secondary_markers', [])
        all_markers = primary_markers + secondary_markers
        
        mentioned_markers = [marker for marker in all_markers if marker in claim_lower]
        
        if mentioned_markers:
            return TestValidationResult(
                validation_id=f"biomarker_claim_accurate_{int(time.time())}",
                test_name="biomarker_claim_validation",
                validation_type=TestValidationType.DOMAIN_ACCURACY,
                severity=ValidationSeverity.INFO,
                passed=True,
                confidence=0.8,
                message=f"Accurate biomarker claim: {mentioned_markers}",
                evidence=[claim],
                metadata={'validated_markers': mentioned_markers}
            )
        else:
            return TestValidationResult(
                validation_id=f"biomarker_claim_unverified_{int(time.time())}",
                test_name="biomarker_claim_validation",
                validation_type=TestValidationType.DOMAIN_ACCURACY,
                severity=ValidationSeverity.MINOR,
                passed=False,
                confidence=0.6,
                message="Unverified biomarker claim",
                evidence=[claim],
                recommendations=["Verify biomarker claim against established disease panels"]
            )
    
    def _extract_pathway_mentions(self, response: str) -> List[str]:
        """Extract pathway mentions from response."""
        pathway_terms = [
            'glycolysis', 'gluconeogenesis', 'tca cycle', 'citric acid cycle',
            'pentose phosphate pathway', 'fatty acid oxidation', 'cholesterol metabolism',
            'amino acid metabolism', 'urea cycle'
        ]
        
        mentioned_pathways = []
        response_lower = response.lower()
        
        for pathway in pathway_terms:
            if pathway in response_lower:
                mentioned_pathways.append(pathway)
        
        return mentioned_pathways
    
    def _validate_pathway_metabolite_associations(self, 
                                                pathway: str, 
                                                response: str) -> List[TestValidationResult]:
        """Validate pathway-metabolite associations."""
        validations = []
        
        # Define pathway-metabolite associations
        pathway_metabolites = {
            'glycolysis': ['glucose', 'lactate', 'pyruvate'],
            'tca cycle': ['acetate', 'citrate'],
            'cholesterol metabolism': ['cholesterol'],
            'urea cycle': ['urea', 'ammonia'],
            'amino acid metabolism': ['alanine', 'glutamine']
        }
        
        if pathway in pathway_metabolites:
            expected_metabolites = pathway_metabolites[pathway]
            mentioned_metabolites = self._extract_mentioned_metabolites(response)
            
            associated_metabolites = [m for m in expected_metabolites if m in mentioned_metabolites]
            
            if associated_metabolites:
                validations.append(TestValidationResult(
                    validation_id=f"pathway_metabolite_{pathway}_{int(time.time())}",
                    test_name="pathway_metabolite_validation",
                    validation_type=TestValidationType.DOMAIN_ACCURACY,
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    confidence=0.8,
                    message=f"Correct pathway-metabolite associations for {pathway}",
                    metadata={
                        'pathway': pathway,
                        'associated_metabolites': associated_metabolites
                    }
                ))
        
        return validations
    
    def _validate_concentration_range(self, 
                                    metabolite: str, 
                                    concentrations: List[float],
                                    expected_ranges: Optional[Dict[str, Tuple[float, float]]]) -> List[TestValidationResult]:
        """Validate metabolite concentration ranges."""
        validations = []
        
        if not concentrations:
            return validations
        
        # Use expected ranges or default metabolite knowledge
        if expected_ranges and metabolite in expected_ranges:
            expected_range = expected_ranges[metabolite]
        elif metabolite in self.metabolite_knowledge:
            metabolite_data = self.metabolite_knowledge[metabolite]
            # Handle both dict and MetaboliteData object
            if hasattr(metabolite_data, 'concentration_range'):
                expected_range = getattr(metabolite_data, 'concentration_range', None)
            elif isinstance(metabolite_data, dict):
                expected_range = metabolite_data.get('concentration_range')
            else:
                expected_range = None
            
            if expected_range is None:
                return validations
        else:
            return validations
        
        min_conc, max_conc = min(concentrations), max(concentrations)
        expected_min, expected_max = expected_range
        
        # Check if concentrations are within reasonable bounds
        reasonable_lower = expected_min * 0.1  # 10x below normal
        reasonable_upper = expected_max * 10   # 10x above normal
        
        if min_conc < reasonable_lower or max_conc > reasonable_upper:
            validations.append(TestValidationResult(
                validation_id=f"concentration_range_{metabolite}_{int(time.time())}",
                test_name="concentration_range_validation",
                validation_type=TestValidationType.DOMAIN_ACCURACY,
                severity=ValidationSeverity.MAJOR,
                passed=False,
                confidence=0.8,
                message=f"Unreasonable concentration range for {metabolite}",
                actual_value=(min_conc, max_conc),
                expected_value=(reasonable_lower, reasonable_upper),
                recommendations=[f"Review concentration data for {metabolite}"]
            ))
        
        return validations
    
    def _validate_concentration_statistics(self, 
                                         metabolite: str, 
                                         concentrations: List[float]) -> List[TestValidationResult]:
        """Validate statistical properties of concentration data."""
        validations = []
        
        if len(concentrations) < 3:
            return validations
        
        # Calculate CV (coefficient of variation)
        mean_conc = statistics.mean(concentrations)
        std_conc = statistics.stdev(concentrations)
        cv = std_conc / mean_conc * 100 if mean_conc > 0 else 0
        
        # Typical biological CV should be < 50%
        if cv > 50:
            validations.append(TestValidationResult(
                validation_id=f"concentration_cv_{metabolite}_{int(time.time())}",
                test_name="concentration_statistics_validation",
                validation_type=TestValidationType.STATISTICAL_CONSISTENCY,
                severity=ValidationSeverity.MINOR,
                passed=False,
                confidence=0.7,
                message=f"High coefficient of variation for {metabolite}: {cv:.1f}%",
                actual_value=cv,
                expected_value=50.0,
                recommendations=[f"Review data quality for {metabolite} concentrations"]
            ))
        
        return validations


# =====================================================================
# VALIDATION REPORT GENERATOR
# =====================================================================

class ValidationReportGenerator:
    """
    Generates comprehensive validation reports with detailed diagnostics.
    """
    
    def __init__(self, output_directory: Optional[Path] = None):
        self.output_directory = output_directory or Path("validation_reports")
        self.output_directory.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_report(self, 
                                    validation_results: List[TestValidationResult],
                                    test_metadata: Dict[str, Any],
                                    report_name: str = "validation_report") -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        timestamp = datetime.now().isoformat()
        
        # Analyze validation results
        analysis = self._analyze_validation_results(validation_results)
        
        # Generate summary statistics
        summary = self._generate_summary_statistics(validation_results)
        
        # Generate recommendations
        recommendations = self._generate_actionable_recommendations(validation_results, analysis)
        
        # Create comprehensive report
        report = {
            'report_metadata': {
                'report_name': report_name,
                'generation_timestamp': timestamp,
                'total_validations': len(validation_results),
                'test_metadata': test_metadata,
                'report_version': '1.0.0'
            },
            'executive_summary': {
                'overall_pass_rate': summary['pass_rate'],
                'critical_issues': summary['critical_count'],
                'major_issues': summary['major_count'],
                'minor_issues': summary['minor_count'],
                'total_recommendations': len(recommendations),
                'validation_confidence_avg': summary['avg_confidence']
            },
            'detailed_analysis': analysis,
            'validation_results': [result.to_dict() for result in validation_results],
            'summary_statistics': summary,
            'recommendations': recommendations,
            'diagnostics': self._generate_diagnostics(validation_results, analysis)
        }
        
        # Save report
        self._save_report(report, report_name)
        
        return report
    
    def generate_cross_validation_report(self, 
                                       cross_validation_results: List[CrossTestValidationResult],
                                       report_name: str = "cross_validation_report") -> Dict[str, Any]:
        """Generate cross-validation report."""
        
        timestamp = datetime.now().isoformat()
        
        # Analyze cross-validation results
        overall_consistency_score = statistics.mean([r.consistency_score for r in cross_validation_results])
        total_inconsistencies = sum(len(r.inconsistencies_found) for r in cross_validation_results)
        all_recommendations = []
        for result in cross_validation_results:
            all_recommendations.extend(result.recommendations)
        
        report = {
            'report_metadata': {
                'report_name': report_name,
                'generation_timestamp': timestamp,
                'cross_validation_suites': len(cross_validation_results),
                'report_version': '1.0.0'
            },
            'executive_summary': {
                'overall_consistency_score': overall_consistency_score,
                'total_inconsistencies': total_inconsistencies,
                'validation_suites_count': len(cross_validation_results),
                'total_recommendations': len(set(all_recommendations))
            },
            'cross_validation_results': [asdict(result) for result in cross_validation_results],
            'consistency_analysis': self._analyze_consistency_patterns(cross_validation_results),
            'recommendations': list(set(all_recommendations))[:15]  # Top 15 unique recommendations
        }
        
        # Save report
        self._save_report(report, report_name)
        
        return report
    
    def generate_trend_analysis_report(self, 
                                     historical_validations: List[List[TestValidationResult]],
                                     time_points: List[datetime],
                                     report_name: str = "trend_analysis_report") -> Dict[str, Any]:
        """Generate trend analysis report for validation results over time."""
        
        timestamp = datetime.now().isoformat()
        
        # Analyze trends
        trend_analysis = self._analyze_validation_trends(historical_validations, time_points)
        
        report = {
            'report_metadata': {
                'report_name': report_name,
                'generation_timestamp': timestamp,
                'time_points_analyzed': len(time_points),
                'report_version': '1.0.0'
            },
            'trend_analysis': trend_analysis,
            'quality_metrics_over_time': self._calculate_quality_metrics_over_time(
                historical_validations, time_points
            ),
            'regression_detection': self._detect_quality_regressions(
                historical_validations, time_points
            ),
            'improvement_opportunities': self._identify_improvement_opportunities(
                trend_analysis
            )
        }
        
        # Save report
        self._save_report(report, report_name)
        
        return report
    
    def _analyze_validation_results(self, 
                                  validation_results: List[TestValidationResult]) -> Dict[str, Any]:
        """Analyze validation results for patterns and insights."""
        
        analysis = {
            'severity_distribution': defaultdict(int),
            'validation_type_distribution': defaultdict(int),
            'test_performance': defaultdict(list),
            'confidence_analysis': {},
            'failure_patterns': []
        }
        
        # Analyze severity and type distributions
        for result in validation_results:
            analysis['severity_distribution'][result.severity.value] += 1
            analysis['validation_type_distribution'][result.validation_type.value] += 1
            analysis['test_performance'][result.test_name].append({
                'passed': result.passed,
                'confidence': result.confidence,
                'severity': result.severity.value
            })
        
        # Analyze confidence levels
        all_confidences = [r.confidence for r in validation_results]
        if all_confidences:
            analysis['confidence_analysis'] = {
                'mean_confidence': statistics.mean(all_confidences),
                'median_confidence': statistics.median(all_confidences),
                'min_confidence': min(all_confidences),
                'max_confidence': max(all_confidences),
                'low_confidence_count': sum(1 for c in all_confidences if c < 0.7)
            }
        
        # Identify failure patterns
        failed_results = [r for r in validation_results if not r.passed]
        if failed_results:
            failure_types = defaultdict(int)
            for result in failed_results:
                failure_types[result.validation_type.value] += 1
            
            analysis['failure_patterns'] = [
                {
                    'failure_type': failure_type,
                    'count': count,
                    'percentage': count / len(failed_results) * 100
                }
                for failure_type, count in failure_types.items()
            ]
        
        return analysis
    
    def _generate_summary_statistics(self, 
                                   validation_results: List[TestValidationResult]) -> Dict[str, Any]:
        """Generate summary statistics for validation results."""
        
        if not validation_results:
            return {
                'total_validations': 0,
                'passed_validations': 0,
                'failed_validations': 0,
                'pass_rate': 0.0,
                'critical_count': 0,
                'major_count': 0,
                'minor_count': 0,
                'warning_count': 0,
                'info_count': 0,
                'avg_confidence': 0.0
            }
        
        passed = sum(1 for r in validation_results if r.passed)
        failed = len(validation_results) - passed
        
        severity_counts = {
            'critical_count': sum(1 for r in validation_results if r.severity == ValidationSeverity.CRITICAL),
            'major_count': sum(1 for r in validation_results if r.severity == ValidationSeverity.MAJOR),
            'minor_count': sum(1 for r in validation_results if r.severity == ValidationSeverity.MINOR),
            'warning_count': sum(1 for r in validation_results if r.severity == ValidationSeverity.WARNING),
            'info_count': sum(1 for r in validation_results if r.severity == ValidationSeverity.INFO)
        }
        
        return {
            'total_validations': len(validation_results),
            'passed_validations': passed,
            'failed_validations': failed,
            'pass_rate': passed / len(validation_results),
            'avg_confidence': statistics.mean([r.confidence for r in validation_results]),
            **severity_counts
        }
    
    def _generate_actionable_recommendations(self, 
                                           validation_results: List[TestValidationResult],
                                           analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on validation results."""
        
        recommendations = []
        
        # Critical issues first
        critical_results = [r for r in validation_results 
                          if r.severity == ValidationSeverity.CRITICAL and not r.passed]
        
        if critical_results:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Safety',
                'title': f"Address {len(critical_results)} critical safety issues",
                'description': "Critical validation failures detected that may impact clinical safety",
                'actions': [
                    "Review all critical validation failures immediately",
                    "Implement fixes before clinical deployment",
                    "Add additional safety validation checks"
                ],
                'affected_tests': list(set(r.test_name for r in critical_results))
            })
        
        # High failure rates
        if analysis.get('failure_patterns'):
            top_failure_type = max(analysis['failure_patterns'], key=lambda x: x['count'])
            if top_failure_type['percentage'] > 30:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Quality',
                    'title': f"Address frequent {top_failure_type['failure_type']} failures",
                    'description': f"{top_failure_type['failure_type']} validation failures represent {top_failure_type['percentage']:.1f}% of all failures",
                    'actions': [
                        f"Focus on improving {top_failure_type['failure_type']} validation",
                        "Review test implementation and validation criteria",
                        "Consider additional training or documentation"
                    ]
                })
        
        # Low confidence validations
        confidence_analysis = analysis.get('confidence_analysis', {})
        if confidence_analysis.get('low_confidence_count', 0) > len(validation_results) * 0.2:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Validation Quality',
                'title': "Improve validation confidence levels",
                'description': f"{confidence_analysis['low_confidence_count']} validations have low confidence (<70%)",
                'actions': [
                    "Review validation criteria and thresholds",
                    "Enhance validation logic for better confidence",
                    "Consider additional validation data sources"
                ]
            })
        
        # Test-specific recommendations
        test_performance = analysis.get('test_performance', {})
        for test_name, performance_data in test_performance.items():
            failed_count = sum(1 for p in performance_data if not p['passed'])
            if failed_count > len(performance_data) * 0.5:  # More than 50% failure rate
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Test Quality',
                    'title': f"Improve {test_name} test reliability",
                    'description': f"Test {test_name} has {failed_count}/{len(performance_data)} failures",
                    'actions': [
                        f"Review and refactor {test_name} test implementation",
                        "Analyze failure root causes",
                        "Consider test environment factors"
                    ]
                })
        
        return recommendations[:10]  # Limit to top 10
    
    def _generate_diagnostics(self, 
                            validation_results: List[TestValidationResult],
                            analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed diagnostics information."""
        
        diagnostics = {
            'validation_health_score': self._calculate_validation_health_score(validation_results),
            'quality_indicators': self._calculate_quality_indicators(validation_results),
            'risk_assessment': self._assess_validation_risks(validation_results),
            'improvement_metrics': self._calculate_improvement_metrics(validation_results)
        }
        
        return diagnostics
    
    def _analyze_consistency_patterns(self, 
                                    cross_validation_results: List[CrossTestValidationResult]) -> Dict[str, Any]:
        """Analyze consistency patterns across cross-validation results."""
        
        inconsistency_types = defaultdict(int)
        all_inconsistencies = []
        
        for result in cross_validation_results:
            all_inconsistencies.extend(result.inconsistencies_found)
        
        for inconsistency in all_inconsistencies:
            inconsistency_types[inconsistency.get('type', 'unknown')] += 1
        
        return {
            'total_inconsistencies': len(all_inconsistencies),
            'inconsistency_type_distribution': dict(inconsistency_types),
            'most_common_inconsistency': max(inconsistency_types.items(), key=lambda x: x[1]) if inconsistency_types else None,
            'consistency_score_distribution': [r.consistency_score for r in cross_validation_results]
        }
    
    def _analyze_validation_trends(self, 
                                 historical_validations: List[List[TestValidationResult]],
                                 time_points: List[datetime]) -> Dict[str, Any]:
        """Analyze trends in validation results over time."""
        
        trends = {
            'pass_rate_trend': [],
            'confidence_trend': [],
            'severity_trends': defaultdict(list),
            'validation_type_trends': defaultdict(list)
        }
        
        for validations in historical_validations:
            if validations:
                # Calculate pass rate
                pass_rate = sum(1 for v in validations if v.passed) / len(validations)
                trends['pass_rate_trend'].append(pass_rate)
                
                # Calculate average confidence
                avg_confidence = statistics.mean([v.confidence for v in validations])
                trends['confidence_trend'].append(avg_confidence)
                
                # Track severity distributions
                severity_counts = defaultdict(int)
                for v in validations:
                    severity_counts[v.severity.value] += 1
                
                for severity in ValidationSeverity:
                    trends['severity_trends'][severity.value].append(
                        severity_counts[severity.value] / len(validations)
                    )
        
        return dict(trends)
    
    def _calculate_quality_metrics_over_time(self, 
                                           historical_validations: List[List[TestValidationResult]],
                                           time_points: List[datetime]) -> List[Dict[str, Any]]:
        """Calculate quality metrics over time."""
        
        metrics_over_time = []
        
        for i, validations in enumerate(historical_validations):
            if i < len(time_points) and validations:
                summary = self._generate_summary_statistics(validations)
                
                metrics_over_time.append({
                    'timestamp': time_points[i].isoformat(),
                    'pass_rate': summary['pass_rate'],
                    'total_validations': summary['total_validations'],
                    'critical_count': summary['critical_count'],
                    'major_count': summary['major_count'],
                    'avg_confidence': summary['avg_confidence']
                })
        
        return metrics_over_time
    
    def _detect_quality_regressions(self, 
                                  historical_validations: List[List[TestValidationResult]],
                                  time_points: List[datetime]) -> List[Dict[str, Any]]:
        """Detect quality regressions over time."""
        
        regressions = []
        
        if len(historical_validations) < 2:
            return regressions
        
        # Calculate pass rates over time
        pass_rates = []
        for validations in historical_validations:
            if validations:
                pass_rate = sum(1 for v in validations if v.passed) / len(validations)
                pass_rates.append(pass_rate)
            else:
                pass_rates.append(0.0)
        
        # Detect significant drops
        for i in range(1, len(pass_rates)):
            current_rate = pass_rates[i]
            previous_rate = pass_rates[i-1]
            
            # Significant regression: drop of more than 20%
            if previous_rate > 0 and (previous_rate - current_rate) / previous_rate > 0.2:
                regressions.append({
                    'type': 'pass_rate_regression',
                    'timestamp': time_points[i].isoformat() if i < len(time_points) else 'unknown',
                    'previous_pass_rate': previous_rate,
                    'current_pass_rate': current_rate,
                    'regression_magnitude': (previous_rate - current_rate) / previous_rate,
                    'severity': 'major' if (previous_rate - current_rate) / previous_rate > 0.3 else 'minor'
                })
        
        return regressions
    
    def _identify_improvement_opportunities(self, trend_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify improvement opportunities based on trend analysis."""
        
        opportunities = []
        
        # Check pass rate trends
        pass_rate_trend = trend_analysis.get('pass_rate_trend', [])
        if len(pass_rate_trend) >= 3:
            recent_trend = pass_rate_trend[-3:]
            if all(recent_trend[i] <= recent_trend[i+1] for i in range(len(recent_trend)-1)):
                opportunities.append({
                    'type': 'positive_trend',
                    'category': 'pass_rate',
                    'description': 'Pass rate shows consistent improvement trend',
                    'recommendation': 'Continue current improvement efforts'
                })
            elif all(recent_trend[i] >= recent_trend[i+1] for i in range(len(recent_trend)-1)):
                opportunities.append({
                    'type': 'negative_trend',
                    'category': 'pass_rate',
                    'description': 'Pass rate shows declining trend',
                    'recommendation': 'Investigate causes of declining validation performance'
                })
        
        # Check confidence trends
        confidence_trend = trend_analysis.get('confidence_trend', [])
        if confidence_trend and statistics.mean(confidence_trend) < 0.8:
            opportunities.append({
                'type': 'low_confidence',
                'category': 'confidence',
                'description': 'Overall validation confidence is below optimal level',
                'recommendation': 'Review validation criteria and improve confidence scoring'
            })
        
        return opportunities
    
    def _calculate_validation_health_score(self, 
                                         validation_results: List[TestValidationResult]) -> float:
        """Calculate overall validation health score (0-100)."""
        
        if not validation_results:
            return 0.0
        
        # Weight factors for health score
        pass_rate = sum(1 for r in validation_results if r.passed) / len(validation_results)
        avg_confidence = statistics.mean([r.confidence for r in validation_results])
        
        # Penalty for critical issues
        critical_count = sum(1 for r in validation_results if r.severity == ValidationSeverity.CRITICAL and not r.passed)
        critical_penalty = min(0.5, critical_count * 0.1)  # Max 50% penalty
        
        # Calculate health score
        health_score = (pass_rate * 0.6 + avg_confidence * 0.4) * 100
        health_score = max(0.0, health_score - critical_penalty * 100)
        
        return min(100.0, health_score)
    
    def _calculate_quality_indicators(self, 
                                    validation_results: List[TestValidationResult]) -> Dict[str, Any]:
        """Calculate quality indicators."""
        
        if not validation_results:
            return {}
        
        return {
            'completeness': len(validation_results) / max(50, len(validation_results)),  # Assuming 50 is ideal
            'reliability': sum(1 for r in validation_results if r.passed) / len(validation_results),
            'confidence_level': statistics.mean([r.confidence for r in validation_results]),
            'critical_issue_rate': sum(1 for r in validation_results if r.severity == ValidationSeverity.CRITICAL) / len(validation_results),
            'validation_coverage': len(set(r.validation_type for r in validation_results)) / len(TestValidationType)
        }
    
    def _assess_validation_risks(self, 
                               validation_results: List[TestValidationResult]) -> Dict[str, Any]:
        """Assess risks based on validation results."""
        
        risks = {
            'high_risk_factors': [],
            'medium_risk_factors': [],
            'low_risk_factors': [],
            'overall_risk_level': 'low'
        }
        
        # Check for high-risk factors
        critical_failures = sum(1 for r in validation_results if r.severity == ValidationSeverity.CRITICAL and not r.passed)
        if critical_failures > 0:
            risks['high_risk_factors'].append(f"{critical_failures} critical validation failures")
            risks['overall_risk_level'] = 'high'
        
        # Check pass rate
        pass_rate = sum(1 for r in validation_results if r.passed) / len(validation_results) if validation_results else 0
        if pass_rate < 0.7:
            risks['high_risk_factors'].append(f"Low pass rate: {pass_rate:.1%}")
            if risks['overall_risk_level'] != 'high':
                risks['overall_risk_level'] = 'medium'
        elif pass_rate < 0.85:
            risks['medium_risk_factors'].append(f"Moderate pass rate: {pass_rate:.1%}")
        
        # Check confidence levels
        avg_confidence = statistics.mean([r.confidence for r in validation_results]) if validation_results else 0
        if avg_confidence < 0.7:
            risks['medium_risk_factors'].append(f"Low average confidence: {avg_confidence:.1%}")
        
        return risks
    
    def _calculate_improvement_metrics(self, 
                                     validation_results: List[TestValidationResult]) -> Dict[str, Any]:
        """Calculate metrics for tracking improvement."""
        
        if not validation_results:
            return {}
        
        # Group by test name
        test_groups = defaultdict(list)
        for result in validation_results:
            test_groups[result.test_name].append(result)
        
        improvement_metrics = {
            'test_reliability_scores': {},
            'validation_type_performance': {},
            'improvement_targets': []
        }
        
        # Calculate reliability scores for each test
        for test_name, results in test_groups.items():
            pass_rate = sum(1 for r in results if r.passed) / len(results)
            avg_confidence = statistics.mean([r.confidence for r in results])
            reliability_score = pass_rate * 0.7 + avg_confidence * 0.3
            
            improvement_metrics['test_reliability_scores'][test_name] = {
                'reliability_score': reliability_score,
                'pass_rate': pass_rate,
                'avg_confidence': avg_confidence,
                'total_validations': len(results)
            }
            
            if reliability_score < 0.8:
                improvement_metrics['improvement_targets'].append({
                    'test_name': test_name,
                    'current_score': reliability_score,
                    'target_score': 0.8,
                    'improvement_needed': 0.8 - reliability_score
                })
        
        # Calculate validation type performance
        type_groups = defaultdict(list)
        for result in validation_results:
            type_groups[result.validation_type].append(result)
        
        for validation_type, results in type_groups.items():
            pass_rate = sum(1 for r in results if r.passed) / len(results)
            improvement_metrics['validation_type_performance'][validation_type.value] = {
                'pass_rate': pass_rate,
                'total_validations': len(results)
            }
        
        return improvement_metrics
    
    def _save_report(self, report: Dict[str, Any], report_name: str):
        """Save report to file."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{report_name}_{timestamp}.json"
        filepath = self.output_directory / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Validation report saved: {filepath}")
            
            # Also save a human-readable summary
            summary_filename = f"{report_name}_{timestamp}_summary.txt"
            summary_filepath = self.output_directory / summary_filename
            
            with open(summary_filepath, 'w') as f:
                f.write(self._generate_human_readable_summary(report))
            
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")
    
    def _generate_human_readable_summary(self, report: Dict[str, Any]) -> str:
        """Generate human-readable summary of validation report."""
        
        summary_lines = []
        summary_lines.append("CLINICAL METABOLOMICS ORACLE - VALIDATION REPORT SUMMARY")
        summary_lines.append("=" * 60)
        summary_lines.append("")
        
        # Executive summary
        if 'executive_summary' in report:
            exec_summary = report['executive_summary']
            summary_lines.append("EXECUTIVE SUMMARY:")
            summary_lines.append(f"Overall Pass Rate: {exec_summary.get('overall_pass_rate', 0):.1%}")
            summary_lines.append(f"Critical Issues: {exec_summary.get('critical_issues', 0)}")
            summary_lines.append(f"Major Issues: {exec_summary.get('major_issues', 0)}")
            summary_lines.append(f"Minor Issues: {exec_summary.get('minor_issues', 0)}")
            summary_lines.append(f"Average Confidence: {exec_summary.get('validation_confidence_avg', 0):.1%}")
            summary_lines.append("")
        
        # Top recommendations
        if 'recommendations' in report and report['recommendations']:
            summary_lines.append("TOP RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'][:5], 1):
                if isinstance(rec, dict):
                    summary_lines.append(f"{i}. [{rec.get('priority', 'MEDIUM')}] {rec.get('title', rec.get('description', 'No title'))}")
                else:
                    summary_lines.append(f"{i}. {str(rec)}")
            summary_lines.append("")
        
        # Validation health score
        if 'diagnostics' in report:
            diagnostics = report['diagnostics']
            if 'validation_health_score' in diagnostics:
                health_score = diagnostics['validation_health_score']
                summary_lines.append(f"VALIDATION HEALTH SCORE: {health_score:.1f}/100")
                summary_lines.append("")
        
        summary_lines.append(f"Report generated: {report.get('report_metadata', {}).get('generation_timestamp', 'Unknown')}")
        
        return "\n".join(summary_lines)


# =====================================================================
# PYTEST FIXTURES FOR VALIDATION TEST UTILITIES
# =====================================================================

@pytest.fixture
def enhanced_biomedical_validator():
    """Provide enhanced biomedical content validator."""
    return EnhancedBiomedicalContentValidator()


@pytest.fixture
def test_result_validator():
    """Provide test result validator."""
    return TestResultValidator()


@pytest.fixture
def clinical_metabolomics_validator():
    """Provide clinical metabolomics validator."""
    return ClinicalMetabolomicsValidator()


@pytest.fixture
def validation_report_generator(tmp_path):
    """Provide validation report generator with temporary directory."""
    return ValidationReportGenerator(output_directory=tmp_path / "validation_reports")


@pytest.fixture
def sample_test_validation_results():
    """Provide sample test validation results for testing."""
    return [
        TestValidationResult(
            validation_id="test_001",
            test_name="metabolite_accuracy_test",
            validation_type=TestValidationType.DOMAIN_ACCURACY,
            severity=ValidationSeverity.INFO,
            passed=True,
            confidence=0.95,
            message="Glucose molecular formula validation passed",
            actual_value="C6H12O6",
            expected_value="C6H12O6",
            metadata={'metabolite': 'glucose'}
        ),
        TestValidationResult(
            validation_id="test_002",
            test_name="statistical_consistency_test",
            validation_type=TestValidationType.STATISTICAL_CONSISTENCY,
            severity=ValidationSeverity.CRITICAL,
            passed=False,
            confidence=1.0,
            message="Correlation coefficient outside valid range",
            actual_value=2.5,
            expected_value=1.0,
            recommendations=["Correct correlation coefficient to be within [-1, 1]"]
        ),
        TestValidationResult(
            validation_id="test_003",
            test_name="response_quality_test",
            validation_type=TestValidationType.RESPONSE_QUALITY,
            severity=ValidationSeverity.MINOR,
            passed=False,
            confidence=0.7,
            message="Response completeness below threshold",
            actual_value=0.6,
            expected_value=0.7,
            recommendations=["Address all aspects of the query", "Provide more comprehensive information"]
        )
    ]


@pytest.fixture
def validation_test_scenarios():
    """Provide comprehensive validation test scenarios."""
    return {
        'biomedical_accuracy_scenarios': [
            {
                'name': 'accurate_metabolite_info',
                'query': 'What is the molecular formula of glucose?',
                'response': 'Glucose has the molecular formula C6H12O6 and molecular weight of 180.16 Da.',
                'expected_validations': ['molecular_formula', 'molecular_weight'],
                'expected_pass': True
            },
            {
                'name': 'inaccurate_metabolite_info',
                'query': 'What is the molecular formula of glucose?',
                'response': 'Glucose has the molecular formula C6H10O6 and molecular weight of 200 Da.',
                'expected_validations': ['molecular_formula', 'molecular_weight'],
                'expected_pass': False
            }
        ],
        'clinical_safety_scenarios': [
            {
                'name': 'safe_clinical_advice',
                'query': 'How can metabolomics help in diabetes management?',
                'response': 'Studies suggest that metabolomics may help monitor glucose metabolism and guide treatment decisions.',
                'expected_validations': ['clinical_safety'],
                'expected_pass': True
            },
            {
                'name': 'unsafe_clinical_advice',
                'query': 'How can I treat my diabetes?',
                'response': 'Stop taking insulin and use glucose supplements to cure diabetes.',
                'expected_validations': ['clinical_safety'],
                'expected_pass': False
            }
        ],
        'statistical_validation_scenarios': [
            {
                'name': 'valid_statistics',
                'response': 'The correlation was r = 0.85 with p < 0.001, showing excellent diagnostic performance.',
                'statistical_data': {
                    'correlation': {'biomarker_disease': 0.85},
                    'p_values': {'significance_test': 0.001},
                    'auc_values': {'diagnostic_performance': 0.92}
                },
                'expected_pass': True
            },
            {
                'name': 'invalid_statistics',
                'response': 'The correlation was r = 2.5 with p = 0.0, showing perfect results.',
                'statistical_data': {
                    'correlation': {'biomarker_disease': 2.5},
                    'p_values': {'significance_test': 0.0}
                },
                'expected_pass': False
            }
        ]
    }


@pytest.fixture
def cross_test_consistency_data():
    """Provide test data for cross-test consistency validation."""
    return [
        {
            'test_name': 'glucose_validation_test_1',
            'result_patterns': {
                'molecular_formula': 'C6H12O6',
                'molecular_weight': 180.16,
                'normal_range': (3.9, 6.1)
            },
            'statistical_data': {
                'correlation': {'glucose_diabetes': 0.85},
                'p_values': {'association_test': 0.001}
            },
            'performance_metrics': {
                'response_time_ms': 150,
                'memory_usage_mb': 45,
                'throughput_ops_per_sec': 6.7
            }
        },
        {
            'test_name': 'glucose_validation_test_2',
            'result_patterns': {
                'molecular_formula': 'C6H12O6',
                'molecular_weight': 180.16,
                'normal_range': (3.8, 6.2)  # Slightly different range
            },
            'statistical_data': {
                'correlation': {'glucose_diabetes': 0.87},  # Slightly different
                'p_values': {'association_test': 0.002}
            },
            'performance_metrics': {
                'response_time_ms': 180,  # Different performance
                'memory_usage_mb': 52,
                'throughput_ops_per_sec': 5.8
            }
        }
    ]


@pytest.fixture
async def async_test_coordinator():
    """Provide async test coordinator for complex validation scenarios."""
    
    class AsyncTestCoordinator:
        """Coordinates complex async validation testing."""
        
        def __init__(self):
            self.active_tests = []
            self.validation_queue = asyncio.Queue()
            self.results_cache = {}
        
        async def run_concurrent_validations(self, 
                                           validation_tasks: List[Callable],
                                           max_concurrent: int = 5) -> List[Any]:
            """Run validation tasks concurrently."""
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def run_with_semaphore(task):
                async with semaphore:
                    return await task()
            
            tasks = [run_with_semaphore(task) for task in validation_tasks]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        async def validate_with_timeout(self, 
                                      validation_func: Callable,
                                      timeout_seconds: float = 30.0) -> Any:
            """Run validation with timeout."""
            try:
                return await asyncio.wait_for(validation_func(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                return TestValidationResult(
                    validation_id=f"timeout_{int(time.time())}",
                    test_name="timeout_validation",
                    validation_type=TestValidationType.PERFORMANCE_VALIDATION,
                    severity=ValidationSeverity.MAJOR,
                    passed=False,
                    confidence=1.0,
                    message=f"Validation timed out after {timeout_seconds} seconds"
                )
        
        def cleanup(self):
            """Cleanup coordinator resources."""
            self.active_tests.clear()
            self.results_cache.clear()
    
    coordinator = AsyncTestCoordinator()
    yield coordinator
    coordinator.cleanup()


# Make key classes available at module level
__all__ = [
    # Core validation classes
    'EnhancedBiomedicalContentValidator',
    'TestResultValidator', 
    'ClinicalMetabolomicsValidator',
    'ValidationReportGenerator',
    
    # Data classes
    'TestValidationResult',
    'CrossTestValidationResult',
    'ValidationAssertion',
    
    # Enums
    'TestValidationType',
    'ValidationSeverity'
]