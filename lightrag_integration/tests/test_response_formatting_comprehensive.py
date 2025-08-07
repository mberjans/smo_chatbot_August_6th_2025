#!/usr/bin/env python3
"""
Comprehensive test suite for response formatting functionality in Clinical Metabolomics RAG.

This test suite provides complete coverage for:
1. BiomedicalResponseFormatter class with entity extraction and content formatting
2. ResponseValidator with quality control metrics and validation
3. Scientific accuracy validation and citation processing
4. Structured response formatting with multiple output formats
5. Integration testing and performance validation

Test Coverage Goals:
- >90% code coverage for all formatting functionality
- All public methods of BiomedicalResponseFormatter and ResponseValidator
- All configuration combinations and output formats
- Error handling and edge cases
- Performance testing for formatting operations
"""

import pytest
import asyncio
import json
import time
import re
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import logging

# Test imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from lightrag_integration.clinical_metabolomics_rag import (
    BiomedicalResponseFormatter,
    ResponseValidator
)


# Test Fixtures and Mock Data
class TestDataProvider:
    """Provider for realistic biomedical test data."""
    
    @staticmethod
    def get_sample_biomedical_response() -> str:
        """Get a sample biomedical response with various content types."""
        return """
        Introduction: Metabolomics analysis of type 2 diabetes mellitus reveals significant metabolic alterations.
        
        Key Findings:
        - Glucose concentrations were elevated to 15.2 ± 3.4 mmol/L (p < 0.001, n=156)
        - Insulin resistance markers showed 2.3-fold increase (95% CI: 1.8-2.9)
        - Glycolysis pathway disruption observed with reduced pyruvate levels
        - Hexokinase enzyme activity decreased by 45% in muscle tissue samples
        - Lactate/pyruvate ratio increased significantly (p = 0.003)
        
        Metabolic Pathways:
        The tricarboxylic acid (TCA) cycle showed marked perturbations with:
        - Citrate synthase activity reduced by 30%
        - Alpha-ketoglutarate dehydrogenase complex dysfunction
        - Succinate accumulation in plasma (3.2 ± 0.8 μmol/L)
        
        Clinical Correlations:
        These metabolic changes correlate with cardiovascular risk factors including:
        - Elevated HbA1c levels (8.4 ± 1.2%)
        - Increased inflammatory markers (CRP > 3.0 mg/L)
        - Reduced insulin sensitivity index (ISI = 2.1 ± 0.6)
        
        Biomarker Potential:
        Several metabolites show promise as diagnostic biomarkers:
        - 1,5-anhydroglucitol (sensitivity: 87%, specificity: 82%)
        - 2-hydroxybutyrate (AUC = 0.84, p < 0.001)
        - Branched-chain amino acids (BCAA) ratio altered
        
        Therapeutic Implications:
        The identified metabolic disruptions suggest potential therapeutic targets:
        - SGLT2 inhibitor therapy may improve glucose handling
        - Metformin effects on hepatic gluconeogenesis
        - Lifestyle interventions targeting glycemic control
        
        Study Limitations:
        This analysis has several limitations including small sample size (n=156),
        cross-sectional design, and potential confounding by medication use.
        
        References:
        1. Smith et al. (2023). Metabolomics in diabetes. Nature Medicine 29:123-135. DOI: 10.1038/s41591-023-02234-x
        2. Johnson A, Brown B. Glycolysis disruption in T2DM. Diabetes Care. 2022;45(8):1234-1245. PMID: 35123456
        3. Lee KH, Park JW. Clinical metabolomics applications. Cell Metab. 2023;37(4):789-802. PMC9876543
        
        Conclusion:
        These findings provide new insights into the metabolic pathophysiology of type 2 diabetes
        and identify potential biomarkers for improved diagnosis and monitoring.
        """
    
    @staticmethod
    def get_problematic_response() -> str:
        """Get a response with various issues for testing validation."""
        return """
        Diabetes always causes complete metabolic failure in 100% of cases.
        Every single patient will definitely die within exactly 30 days.
        Glucose levels are precisely 999999 mg/L in all patients without exception.
        This treatment completely eliminates all symptoms forever with zero side effects.
        No further research is ever needed because everything is perfectly understood.
        All doctors agree unanimously on every aspect of treatment.
        The cure rate is exactly 100% with no failures ever reported.
        """
    
    @staticmethod
    def get_sample_metadata() -> Dict[str, Any]:
        """Get sample metadata for testing."""
        return {
            'mode': 'hybrid',
            'processing_time': 2.3,
            'sources': [
                {
                    'title': 'Metabolomics in Type 2 Diabetes',
                    'authors': ['Smith, J.', 'Doe, A.'],
                    'journal': 'Nature Medicine',
                    'year': 2023,
                    'doi': '10.1038/s41591-023-02234-x',
                    'pmid': '37123456'
                }
            ],
            'confidence': 0.85,
            'query_type': 'clinical_analysis'
        }
    
    @staticmethod
    def get_statistical_response() -> str:
        """Get a response rich in statistical content."""
        return """
        Statistical Analysis Results:
        
        Primary Outcomes:
        - Mean difference: 12.5 ± 2.3 (p < 0.001, 95% CI: 8.0-17.0)
        - Effect size: Cohen's d = 1.2 (large effect)
        - Statistical power: 95% (β = 0.05)
        
        Secondary Analysis:
        - Correlation coefficient: r = 0.78 (p < 0.001)
        - Regression coefficient: β = 2.34 (SE = 0.45, p = 0.003)
        - Odds ratio: OR = 3.2 (95% CI: 1.8-5.7, p = 0.001)
        
        Diagnostic Performance:
        - Sensitivity: 89.2% (95% CI: 84.1-93.4%)
        - Specificity: 76.8% (95% CI: 71.2-81.9%)
        - Positive predictive value: 82.1%
        - Negative predictive value: 85.6%
        - Area under ROC curve: 0.91 (95% CI: 0.87-0.95)
        
        Quality Control:
        - Coefficient of variation: 4.2%
        - Intra-assay precision: CV < 5%
        - Inter-assay precision: CV < 8%
        - Recovery: 98.5 ± 3.2%
        """


@pytest.fixture
def formatter():
    """Create a BiomedicalResponseFormatter instance for testing."""
    return BiomedicalResponseFormatter()


@pytest.fixture
def custom_formatter():
    """Create a formatter with custom configuration."""
    config = {
        'extract_entities': True,
        'format_statistics': True,
        'process_sources': True,
        'max_entity_extraction': 20,
        'validate_scientific_accuracy': True,
        'enhanced_citation_processing': True
    }
    return BiomedicalResponseFormatter(config)


@pytest.fixture
def validator():
    """Create a ResponseValidator instance for testing."""
    return ResponseValidator()


@pytest.fixture
def custom_validator():
    """Create a validator with custom configuration."""
    config = {
        'enabled': True,
        'performance_mode': 'comprehensive',
        'quality_gate_enabled': True,
        'thresholds': {
            'minimum_quality_score': 0.7,
            'scientific_confidence_threshold': 0.8
        }
    }
    return ResponseValidator(config)


@pytest.fixture
def sample_data():
    """Get sample test data."""
    return TestDataProvider()


# BiomedicalResponseFormatter Tests

class TestBiomedicalResponseFormatter:
    """Test suite for BiomedicalResponseFormatter class."""
    
    def test_init_default_config(self):
        """Test formatter initialization with default configuration."""
        formatter = BiomedicalResponseFormatter()
        assert formatter.config is not None
        assert formatter.config['extract_entities'] is True
        assert formatter.config['format_statistics'] is True
        assert formatter.config['process_sources'] is True
    
    def test_init_custom_config(self, custom_formatter):
        """Test formatter initialization with custom configuration."""
        assert custom_formatter.config['max_entity_extraction'] == 20
        assert custom_formatter.config['validate_scientific_accuracy'] is True
    
    def test_format_response_basic(self, formatter, sample_data):
        """Test basic response formatting functionality."""
        raw_response = sample_data.get_sample_biomedical_response()
        metadata = sample_data.get_sample_metadata()
        
        result = formatter.format_response(raw_response, metadata)
        
        # Check basic structure
        assert isinstance(result, dict)
        assert 'formatted_content' in result
        assert 'entities' in result
        assert 'statistics' in result
        assert 'sources' in result
        assert 'formatting_metadata' in result
        
        # Check content is processed
        assert len(result['formatted_content']) > 0
        assert result['formatted_content'] != raw_response
    
    def test_entity_extraction_metabolites(self, formatter, sample_data):
        """Test metabolite entity extraction."""
        raw_response = sample_data.get_sample_biomedical_response()
        result = formatter.format_response(raw_response)
        
        entities = result['entities']
        assert 'metabolites' in entities
        
        metabolites = entities['metabolites']
        expected_metabolites = ['glucose', 'pyruvate', 'lactate', 'citrate', 'succinate']
        
        # Check that some expected metabolites are found
        found_metabolites = [m.lower() for m in metabolites]
        assert any(metabolite in ' '.join(found_metabolites) for metabolite in expected_metabolites)
    
    def test_entity_extraction_proteins(self, formatter, sample_data):
        """Test protein entity extraction."""
        raw_response = sample_data.get_sample_biomedical_response()
        result = formatter.format_response(raw_response)
        
        entities = result['entities']
        assert 'proteins' in entities
        
        proteins = entities['proteins']
        expected_proteins = ['hexokinase', 'citrate synthase']
        
        # Check that some expected proteins are found
        found_proteins = [p.lower() for p in proteins]
        assert any(protein in ' '.join(found_proteins) for protein in expected_proteins)
    
    def test_entity_extraction_pathways(self, formatter, sample_data):
        """Test pathway entity extraction."""
        raw_response = sample_data.get_sample_biomedical_response()
        result = formatter.format_response(raw_response)
        
        entities = result['entities']
        assert 'pathways' in entities
        
        pathways = entities['pathways']
        expected_pathways = ['glycolysis', 'tca cycle']
        
        # Check that some expected pathways are found
        found_pathways = [p.lower() for p in pathways]
        assert any(pathway in ' '.join(found_pathways) for pathway in expected_pathways)
    
    def test_entity_extraction_diseases(self, formatter, sample_data):
        """Test disease entity extraction."""
        raw_response = sample_data.get_sample_biomedical_response()
        result = formatter.format_response(raw_response)
        
        entities = result['entities']
        assert 'diseases' in entities
        
        diseases = entities['diseases']
        expected_diseases = ['diabetes', 'cardiovascular']
        
        # Check that some expected diseases are found
        found_diseases = [d.lower() for d in diseases]
        assert any(disease in ' '.join(found_diseases) for disease in expected_diseases)
    
    def test_statistical_data_extraction(self, formatter, sample_data):
        """Test statistical data extraction and formatting."""
        raw_response = sample_data.get_statistical_response()
        result = formatter.format_response(raw_response)
        
        statistics = result['statistics']
        assert isinstance(statistics, list)
        assert len(statistics) > 0
        
        # Check for different types of statistics
        stat_types = [stat.get('type') for stat in statistics]
        assert 'p_value' in stat_types or any('p' in str(stat.get('text', '')) for stat in statistics)
        
        # Check statistical values are properly extracted
        for stat in statistics[:3]:  # Check first few statistics
            assert 'text' in stat
            assert 'value' in stat or 'raw_text' in stat
    
    def test_source_citation_extraction(self, formatter, sample_data):
        """Test source citation extraction and processing."""
        raw_response = sample_data.get_sample_biomedical_response()
        metadata = sample_data.get_sample_metadata()
        
        result = formatter.format_response(raw_response, metadata)
        
        sources = result['sources']
        assert isinstance(sources, list)
        assert len(sources) > 0
        
        # Check source structure
        for source in sources:
            assert isinstance(source, dict)
            # Should have some identifying information
            assert any(key in source for key in ['title', 'authors', 'doi', 'pmid', 'text'])
    
    def test_clinical_indicators(self, formatter, sample_data):
        """Test clinical indicator generation."""
        raw_response = sample_data.get_sample_biomedical_response()
        result = formatter.format_response(raw_response)
        
        assert 'clinical_indicators' in result
        clinical_indicators = result['clinical_indicators']
        
        assert isinstance(clinical_indicators, dict)
        expected_indicators = ['clinical_utility', 'diagnostic_relevance', 'therapeutic_implications']
        
        for indicator in expected_indicators:
            assert indicator in clinical_indicators or len(clinical_indicators) > 0
    
    def test_formatting_metadata(self, formatter, sample_data):
        """Test formatting metadata generation."""
        raw_response = sample_data.get_sample_biomedical_response()
        result = formatter.format_response(raw_response)
        
        assert 'formatting_metadata' in result
        metadata = result['formatting_metadata']
        
        assert isinstance(metadata, dict)
        assert 'processed_at' in metadata
        assert 'applied_formatting' in metadata
        assert isinstance(metadata['applied_formatting'], list)
    
    def test_empty_response_handling(self, formatter):
        """Test handling of empty responses."""
        result = formatter.format_response("")
        
        assert isinstance(result, dict)
        assert 'formatted_content' in result
        assert result['formatted_content'] == ""
        assert 'entities' in result
        assert 'statistics' in result
        assert 'sources' in result
    
    def test_none_response_handling(self, formatter):
        """Test handling of None responses."""
        result = formatter.format_response(None)
        
        assert isinstance(result, dict)
        assert result['formatted_content'] == ""
    
    def test_large_response_handling(self, formatter):
        """Test handling of very large responses."""
        large_response = "Test content. " * 10000  # Very large response
        
        start_time = time.time()
        result = formatter.format_response(large_response)
        processing_time = time.time() - start_time
        
        assert isinstance(result, dict)
        assert processing_time < 30.0  # Should complete within 30 seconds
    
    def test_configuration_disable_entities(self):
        """Test disabling entity extraction via configuration."""
        config = {'extract_entities': False}
        formatter = BiomedicalResponseFormatter(config)
        
        raw_response = TestDataProvider.get_sample_biomedical_response()
        result = formatter.format_response(raw_response)
        
        # Entities should be empty or minimal when disabled
        entities = result.get('entities', {})
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        assert total_entities == 0 or not config['extract_entities']
    
    def test_configuration_disable_statistics(self):
        """Test disabling statistical formatting via configuration."""
        config = {'format_statistics': False}
        formatter = BiomedicalResponseFormatter(config)
        
        raw_response = TestDataProvider.get_statistical_response()
        result = formatter.format_response(raw_response)
        
        # Statistics should be empty or minimal when disabled
        statistics = result.get('statistics', [])
        assert len(statistics) == 0 or not config['format_statistics']
    
    def test_max_entity_limit(self):
        """Test entity extraction limit configuration."""
        config = {'max_entity_extraction': 5}
        formatter = BiomedicalResponseFormatter(config)
        
        raw_response = TestDataProvider.get_sample_biomedical_response()
        result = formatter.format_response(raw_response)
        
        entities = result['entities']
        for entity_type, entity_list in entities.items():
            assert len(entity_list) <= config['max_entity_extraction']


# ResponseValidator Tests

class TestResponseValidator:
    """Test suite for ResponseValidator class."""
    
    def test_init_default_config(self):
        """Test validator initialization with default configuration."""
        validator = ResponseValidator()
        assert validator.config is not None
        assert hasattr(validator, 'quality_weights')
        assert hasattr(validator, 'thresholds')
    
    def test_init_custom_config(self, custom_validator):
        """Test validator initialization with custom configuration."""
        assert custom_validator.config['performance_mode'] == 'comprehensive'
        assert custom_validator.thresholds['minimum_quality_score'] == 0.7
    
    @pytest.mark.asyncio
    async def test_validate_response_basic(self, validator, sample_data):
        """Test basic response validation functionality."""
        response = sample_data.get_sample_biomedical_response()
        query = "What are the metabolic changes in type 2 diabetes?"
        metadata = sample_data.get_sample_metadata()
        
        result = await validator.validate_response(response, query, metadata)
        
        # Check basic structure
        assert isinstance(result, dict)
        assert 'validation_passed' in result
        assert 'quality_score' in result
        assert 'validation_results' in result
        assert 'recommendations' in result
        
        # Check quality score is valid
        assert 0 <= result['quality_score'] <= 1
    
    @pytest.mark.asyncio
    async def test_validate_good_response(self, validator, sample_data):
        """Test validation of a high-quality biomedical response."""
        response = sample_data.get_sample_biomedical_response()
        query = "What are the metabolic changes in type 2 diabetes?"
        metadata = sample_data.get_sample_metadata()
        
        result = await validator.validate_response(response, query, metadata)
        
        # Good response should pass validation
        assert result['validation_passed'] is True
        assert result['quality_score'] > 0.5
    
    @pytest.mark.asyncio
    async def test_validate_problematic_response(self, validator, sample_data):
        """Test validation of a problematic response."""
        response = sample_data.get_problematic_response()
        query = "What are the metabolic changes in type 2 diabetes?"
        metadata = sample_data.get_sample_metadata()
        
        result = await validator.validate_response(response, query, metadata)
        
        # Problematic response should fail validation
        assert result['validation_passed'] is False
        assert result['quality_score'] < 0.7
        
        # Should have recommendations for improvement
        assert len(result['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_scientific_accuracy_validation(self, validator, sample_data):
        """Test scientific accuracy validation components."""
        response = sample_data.get_sample_biomedical_response()
        query = "What are the metabolic changes in type 2 diabetes?"
        metadata = {'mode': 'hybrid', 'confidence': 0.8, 'sources': []}
        
        result = await validator.validate_response(response, query, metadata)
        
        validation_results = result['validation_results']
        assert 'scientific_accuracy' in validation_results
        
        accuracy = validation_results['scientific_accuracy']
        assert 'score' in accuracy
        assert 'details' in accuracy
        assert 0 <= accuracy['score'] <= 1
    
    @pytest.mark.asyncio
    async def test_completeness_assessment(self, validator, sample_data):
        """Test response completeness assessment."""
        response = sample_data.get_sample_biomedical_response()
        query = "What are the metabolic changes in type 2 diabetes?"
        metadata = {'mode': 'hybrid', 'confidence': 0.8, 'sources': []}
        
        result = await validator.validate_response(response, query, metadata)
        
        validation_results = result['validation_results']
        assert 'completeness' in validation_results
        
        completeness = validation_results['completeness']
        assert 'score' in completeness
        assert 0 <= completeness['score'] <= 1
    
    @pytest.mark.asyncio
    async def test_clarity_assessment(self, validator, sample_data):
        """Test response clarity assessment."""
        response = sample_data.get_sample_biomedical_response()
        query = "What are the metabolic changes in type 2 diabetes?"
        metadata = {'mode': 'hybrid', 'confidence': 0.8, 'sources': []}
        
        result = await validator.validate_response(response, query, metadata)
        
        validation_results = result['validation_results']
        assert 'clarity' in validation_results
        
        clarity = validation_results['clarity']
        assert 'score' in clarity
        assert 0 <= clarity['score'] <= 1
    
    @pytest.mark.asyncio
    async def test_hallucination_detection(self, validator, sample_data):
        """Test hallucination detection functionality."""
        response = sample_data.get_problematic_response()
        query = "What are the metabolic changes in type 2 diabetes?"
        metadata = {'mode': 'hybrid', 'confidence': 0.8, 'sources': []}
        
        result = await validator.validate_response(response, query, metadata)
        
        validation_results = result['validation_results']
        assert 'hallucination_check' in validation_results
        
        hallucination = validation_results['hallucination_check']
        assert 'risk_level' in hallucination
        assert hallucination['risk_level'] in ['low', 'medium', 'high']
        
        # Problematic response should show high hallucination risk
        assert hallucination['risk_level'] == 'high'
    
    @pytest.mark.asyncio
    async def test_confidence_assessment(self, validator, sample_data):
        """Test confidence assessment and uncertainty quantification."""
        response = sample_data.get_sample_biomedical_response()
        query = "What are the metabolic changes in type 2 diabetes?"
        metadata = {'mode': 'hybrid', 'confidence': 0.8, 'sources': []}
        
        result = await validator.validate_response(response, query, metadata)
        
        validation_results = result['validation_results']
        assert 'confidence_assessment' in validation_results
        
        confidence = validation_results['confidence_assessment']
        assert 'overall_confidence' in confidence
        assert 0 <= confidence['overall_confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_source_credibility(self, validator, sample_data):
        """Test source credibility assessment."""
        response = sample_data.get_sample_biomedical_response()
        query = "What are the metabolic changes in type 2 diabetes?"
        metadata = sample_data.get_sample_metadata()
        
        result = await validator.validate_response(response, query, metadata)
        
        validation_results = result['validation_results']
        if 'source_credibility' in validation_results:
            source_credibility = validation_results['source_credibility']
            assert 'score' in source_credibility
            assert 0 <= source_credibility['score'] <= 1
    
    @pytest.mark.asyncio
    async def test_quality_gate_functionality(self, custom_validator, sample_data):
        """Test quality gate functionality with custom thresholds."""
        response = sample_data.get_problematic_response()
        query = "What are the metabolic changes in type 2 diabetes?"
        
        result = await custom_validator.validate_response(response, query)
        
        # With higher thresholds, problematic response should definitely fail
        assert result['validation_passed'] is False
        assert result['quality_score'] < custom_validator.thresholds['minimum_quality_score']
    
    @pytest.mark.asyncio
    async def test_empty_response_validation(self, validator):
        """Test validation of empty responses."""
        metadata = {'mode': 'hybrid', 'confidence': 0.0, 'sources': []}
        result = await validator.validate_response("", "test query", metadata)
        
        assert isinstance(result, dict)
        assert result['validation_passed'] is False
        assert result['quality_score'] == 0
        assert len(result['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_performance_mode_fast(self):
        """Test fast performance mode configuration."""
        config = {'performance_mode': 'fast'}
        validator = ResponseValidator(config)
        
        response = TestDataProvider.get_sample_biomedical_response()
        query = "Test query"
        metadata = {'mode': 'hybrid', 'confidence': 0.8, 'sources': []}
        
        start_time = time.time()
        result = await validator.validate_response(response, query, metadata)
        processing_time = time.time() - start_time
        
        assert isinstance(result, dict)
        assert processing_time < 5.0  # Fast mode should be quick
    
    @pytest.mark.asyncio
    async def test_performance_mode_comprehensive(self):
        """Test comprehensive performance mode configuration."""
        config = {'performance_mode': 'comprehensive'}
        validator = ResponseValidator(config)
        
        response = TestDataProvider.get_sample_biomedical_response()
        query = "Test query"
        
        metadata = {'mode': 'hybrid', 'confidence': 0.8, 'sources': []}
        result = await validator.validate_response(response, query, metadata)
        
        # Comprehensive mode should provide more detailed results
        validation_results = result['validation_results']
        assert len(validation_results) >= 4  # Should have multiple validation dimensions


# Structured Response Formatting Tests

class TestStructuredResponseFormatting:
    """Test suite for structured response formatting functionality."""
    
    def test_comprehensive_format(self, formatter, sample_data):
        """Test comprehensive output format."""
        raw_response = sample_data.get_sample_biomedical_response()
        metadata = sample_data.get_sample_metadata()
        
        # Test with comprehensive format configuration
        config = {'output_format': 'comprehensive'}
        formatter.config.update(config)
        
        result = formatter.format_response(raw_response, metadata)
        
        # Should include all formatting components
        assert 'formatted_content' in result
        assert 'entities' in result
        assert 'statistics' in result
        assert 'sources' in result
        assert 'clinical_indicators' in result
    
    def test_clinical_report_format(self, formatter, sample_data):
        """Test clinical report output format."""
        raw_response = sample_data.get_sample_biomedical_response()
        metadata = sample_data.get_sample_metadata()
        
        result = formatter.format_response(raw_response, metadata)
        
        # Should be structured for clinical use
        assert isinstance(result, dict)
        assert 'formatted_content' in result
        
        # Check for clinical-specific formatting
        content = result['formatted_content']
        assert isinstance(content, str)
    
    def test_research_summary_format(self, formatter, sample_data):
        """Test research summary output format."""
        raw_response = sample_data.get_sample_biomedical_response()
        
        result = formatter.format_response(raw_response)
        
        # Should be structured for research purposes
        assert isinstance(result, dict)
        assert 'entities' in result
        assert 'statistics' in result
    
    def test_api_friendly_format(self, formatter, sample_data):
        """Test API-friendly output format."""
        raw_response = sample_data.get_sample_biomedical_response()
        
        result = formatter.format_response(raw_response)
        
        # Should be structured for API consumption
        assert isinstance(result, dict)
        
        # All values should be JSON-serializable
        try:
            json.dumps(result)
        except (TypeError, ValueError):
            pytest.fail("Result is not JSON-serializable")
    
    def test_metadata_generation(self, formatter, sample_data):
        """Test metadata generation for structured responses."""
        raw_response = sample_data.get_sample_biomedical_response()
        metadata = sample_data.get_sample_metadata()
        
        result = formatter.format_response(raw_response, metadata)
        
        assert 'formatting_metadata' in result
        formatting_metadata = result['formatting_metadata']
        
        assert isinstance(formatting_metadata, dict)
        assert 'processed_at' in formatting_metadata
        assert 'applied_formatting' in formatting_metadata
    
    def test_semantic_annotations(self, formatter, sample_data):
        """Test semantic annotations in formatted responses."""
        raw_response = sample_data.get_sample_biomedical_response()
        
        result = formatter.format_response(raw_response)
        
        # Check for semantic annotations in entities
        entities = result['entities']
        assert isinstance(entities, dict)
        
        # Entities should be categorized semantically
        expected_categories = ['metabolites', 'proteins', 'pathways', 'diseases']
        for category in expected_categories:
            assert category in entities
    
    def test_hierarchical_content_structure(self, formatter, sample_data):
        """Test hierarchical structure of formatted content."""
        raw_response = sample_data.get_sample_biomedical_response()
        
        result = formatter.format_response(raw_response)
        
        # Check hierarchical structure
        assert isinstance(result, dict)
        
        # Top-level categories should be present
        top_level_keys = ['formatted_content', 'entities', 'statistics', 'sources']
        for key in top_level_keys:
            assert key in result
        
        # Nested structures should be properly formatted
        entities = result['entities']
        assert isinstance(entities, dict)
        
        statistics = result['statistics']
        assert isinstance(statistics, list)


# Integration Tests

class TestIntegrationFormatting:
    """Integration tests for complete formatting pipeline."""
    
    def test_end_to_end_formatting(self, formatter, sample_data):
        """Test complete end-to-end formatting pipeline."""
        raw_response = sample_data.get_sample_biomedical_response()
        metadata = sample_data.get_sample_metadata()
        
        result = formatter.format_response(raw_response, metadata)
        
        # Should have all components
        required_components = [
            'formatted_content',
            'entities',
            'statistics',
            'sources',
            'clinical_indicators',
            'formatting_metadata'
        ]
        
        for component in required_components:
            assert component in result
    
    @pytest.mark.asyncio
    async def test_formatter_validator_integration(self, formatter, validator, sample_data):
        """Test integration between formatter and validator."""
        raw_response = sample_data.get_sample_biomedical_response()
        metadata = sample_data.get_sample_metadata()
        query = "What are the metabolic changes in type 2 diabetes?"
        
        # First format the response
        formatted_result = formatter.format_response(raw_response, metadata)
        
        # Then validate the formatted response
        validation_result = await validator.validate_response(
            formatted_result['formatted_content'], 
            query, 
            metadata
        )
        
        # Both operations should succeed
        assert isinstance(formatted_result, dict)
        assert isinstance(validation_result, dict)
        assert validation_result['validation_passed'] is True
    
    def test_backward_compatibility(self, formatter, sample_data):
        """Test backward compatibility with existing response structure."""
        raw_response = sample_data.get_sample_biomedical_response()
        
        result = formatter.format_response(raw_response)
        
        # Should maintain basic response structure
        assert 'formatted_content' in result
        assert isinstance(result['formatted_content'], str)
        
        # Enhanced features should be additive, not breaking
        assert len(result) >= 4  # Should have additional fields
    
    def test_performance_impact_assessment(self, formatter, sample_data):
        """Test performance impact of formatting enhancements."""
        raw_response = sample_data.get_sample_biomedical_response()
        
        # Test without enhancements
        basic_config = {
            'extract_entities': False,
            'format_statistics': False,
            'process_sources': False,
            'validate_scientific_accuracy': False
        }
        basic_formatter = BiomedicalResponseFormatter(basic_config)
        
        start_time = time.time()
        basic_result = basic_formatter.format_response(raw_response)
        basic_time = time.time() - start_time
        
        # Test with full enhancements
        start_time = time.time()
        enhanced_result = formatter.format_response(raw_response)
        enhanced_time = time.time() - start_time
        
        # Enhanced processing should complete within reasonable time
        assert enhanced_time < 30.0  # Should complete within 30 seconds
        
        # Enhanced result should have more information
        assert len(enhanced_result) > len(basic_result)
    
    def test_mode_specific_configurations(self, sample_data):
        """Test mode-specific formatting configurations."""
        raw_response = sample_data.get_sample_biomedical_response()
        
        # Clinical mode
        clinical_config = {
            'extract_entities': True,
            'add_clinical_indicators': True,
            'highlight_metabolites': True,
            'format_statistics': True
        }
        clinical_formatter = BiomedicalResponseFormatter(clinical_config)
        clinical_result = clinical_formatter.format_response(raw_response)
        
        # Research mode
        research_config = {
            'extract_entities': True,
            'process_sources': True,
            'validate_scientific_accuracy': True,
            'enhanced_citation_processing': True
        }
        research_formatter = BiomedicalResponseFormatter(research_config)
        research_result = research_formatter.format_response(raw_response)
        
        # Both should succeed but have different emphasis
        assert isinstance(clinical_result, dict)
        assert isinstance(research_result, dict)
        
        # Clinical result should emphasize clinical indicators
        assert 'clinical_indicators' in clinical_result
        
        # Research result should emphasize sources and validation
        assert 'sources' in research_result


# Error Handling and Edge Cases

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_malformed_input_handling(self, formatter):
        """Test handling of malformed input data."""
        # Test with various malformed inputs
        malformed_inputs = [
            None,
            "",
            "   ",  # Whitespace only
            "\n\n\n",  # Newlines only
            "a" * 100000,  # Extremely long string
            {"invalid": "dict input"},  # Wrong type
            123,  # Wrong type
            []  # Wrong type
        ]
        
        for malformed_input in malformed_inputs:
            try:
                result = formatter.format_response(malformed_input)
                assert isinstance(result, dict)
                assert 'formatted_content' in result
            except Exception as e:
                # Should not raise unhandled exceptions
                pytest.fail(f"Formatter raised exception for input {type(malformed_input)}: {e}")
    
    def test_invalid_metadata_handling(self, formatter, sample_data):
        """Test handling of invalid metadata."""
        raw_response = sample_data.get_sample_biomedical_response()
        
        invalid_metadata_cases = [
            None,
            "invalid string",
            123,
            [],
            {"malformed": {"nested": {"too": {"deep": "structure"}}}}
        ]
        
        for invalid_metadata in invalid_metadata_cases:
            try:
                result = formatter.format_response(raw_response, invalid_metadata)
                assert isinstance(result, dict)
            except Exception as e:
                pytest.fail(f"Formatter failed with invalid metadata: {e}")
    
    @pytest.mark.asyncio
    async def test_validator_error_handling(self, validator):
        """Test validator error handling with invalid inputs."""
        invalid_inputs = [
            (None, "query"),
            ("response", None),
            ("", ""),
            (123, "query"),
            ("response", 123)
        ]
        
        for response, query in invalid_inputs:
            try:
                metadata = {'mode': 'hybrid', 'confidence': 0.0, 'sources': []}
                result = await validator.validate_response(response, query, metadata)
                assert isinstance(result, dict)
                assert 'validation_passed' in result
            except Exception as e:
                pytest.fail(f"Validator failed with invalid input: {e}")
    
    def test_configuration_validation(self):
        """Test validation of formatter configuration."""
        # Test invalid configurations
        invalid_configs = [
            {"extract_entities": "invalid_boolean"},
            {"max_entity_extraction": "not_a_number"},
            {"max_entity_extraction": -1},
            {"unknown_config_key": True}
        ]
        
        for invalid_config in invalid_configs:
            try:
                formatter = BiomedicalResponseFormatter(invalid_config)
                # Should handle gracefully, not crash
                assert formatter is not None
            except Exception as e:
                # Some validation errors are acceptable
                assert "config" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_memory_management_large_responses(self, formatter):
        """Test memory management with very large responses."""
        # Create a very large response
        large_response = """
        This is a large biomedical response containing many metabolites like glucose, 
        fructose, sucrose, lactose, galactose, ribose, xylose, arabinose, mannose, 
        and numerous proteins, pathways, and statistical data.
        """ * 1000  # Repeat to create large content
        
        try:
            start_time = time.time()
            result = formatter.format_response(large_response)
            processing_time = time.time() - start_time
            
            assert isinstance(result, dict)
            assert processing_time < 60.0  # Should complete within 1 minute
        except MemoryError:
            pytest.skip("System ran out of memory during large response test")


# Performance Tests

class TestPerformanceFormatting:
    """Performance tests for formatting operations."""
    
    def test_entity_extraction_performance(self, formatter, sample_data):
        """Test performance of entity extraction."""
        raw_response = sample_data.get_sample_biomedical_response()
        
        start_time = time.time()
        result = formatter.format_response(raw_response)
        processing_time = time.time() - start_time
        
        # Entity extraction should be reasonably fast
        assert processing_time < 10.0
        
        # Should extract a reasonable number of entities
        entities = result['entities']
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        assert total_entities > 0
    
    def test_statistical_formatting_performance(self, formatter, sample_data):
        """Test performance of statistical data formatting."""
        raw_response = sample_data.get_statistical_response()
        
        start_time = time.time()
        result = formatter.format_response(raw_response)
        processing_time = time.time() - start_time
        
        # Statistical formatting should be fast
        assert processing_time < 5.0
        
        # Should extract statistical information
        statistics = result['statistics']
        assert len(statistics) > 0
    
    @pytest.mark.asyncio
    async def test_validation_performance(self, validator, sample_data):
        """Test performance of validation operations."""
        response = sample_data.get_sample_biomedical_response()
        query = "Test query for performance"
        metadata = {'mode': 'hybrid', 'confidence': 0.8, 'sources': []}
        
        start_time = time.time()
        result = await validator.validate_response(response, query, metadata)
        processing_time = time.time() - start_time
        
        # Validation should complete within reasonable time
        assert processing_time < 15.0
        assert isinstance(result, dict)
    
    def test_concurrent_formatting(self, formatter, sample_data):
        """Test concurrent formatting operations."""
        raw_response = sample_data.get_sample_biomedical_response()
        
        import concurrent.futures
        import threading
        
        def format_response():
            return formatter.format_response(raw_response)
        
        # Test concurrent formatting
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(format_response) for _ in range(3)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All concurrent operations should succeed
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert 'formatted_content' in result
    
    def test_memory_efficiency(self, formatter, sample_data):
        """Test memory efficiency of formatting operations."""
        raw_response = sample_data.get_sample_biomedical_response()
        
        # Run multiple formatting operations to check for memory leaks
        results = []
        for _ in range(10):
            result = formatter.format_response(raw_response)
            results.append(result)
        
        # Should complete without memory issues
        assert len(results) == 10
        
        # All results should be valid
        for result in results:
            assert isinstance(result, dict)


# Configuration Management Tests

class TestConfigurationManagement:
    """Test configuration management and validation."""
    
    def test_default_configuration_loading(self):
        """Test loading of default configuration."""
        formatter = BiomedicalResponseFormatter()
        
        # Default config should be loaded
        assert formatter.config is not None
        assert isinstance(formatter.config, dict)
        
        # Should have expected default values
        expected_defaults = {
            'extract_entities': True,
            'format_statistics': True,
            'process_sources': True,
            'structure_sections': True
        }
        
        for key, expected_value in expected_defaults.items():
            assert formatter.config.get(key) == expected_value
    
    def test_custom_configuration_override(self):
        """Test custom configuration overriding defaults."""
        custom_config = {
            'extract_entities': False,
            'max_entity_extraction': 10,
            'custom_option': True
        }
        
        formatter = BiomedicalResponseFormatter(custom_config)
        
        # Custom values should override defaults
        assert formatter.config['extract_entities'] is False
        assert formatter.config['max_entity_extraction'] == 10
        assert formatter.config.get('custom_option') is True
        
        # Non-overridden defaults should remain
        assert formatter.config['format_statistics'] is True
    
    def test_configuration_validation_types(self):
        """Test configuration value type validation."""
        # Test boolean configurations
        bool_configs = [
            ('extract_entities', True),
            ('extract_entities', False),
            ('format_statistics', True),
            ('process_sources', False)
        ]
        
        for config_key, config_value in bool_configs:
            config = {config_key: config_value}
            formatter = BiomedicalResponseFormatter(config)
            assert formatter.config[config_key] == config_value
    
    def test_validator_configuration(self):
        """Test validator configuration management."""
        custom_config = {
            'enabled': True,
            'performance_mode': 'fast',
            'thresholds': {
                'minimum_quality_score': 0.8
            }
        }
        
        validator = ResponseValidator(custom_config)
        
        # Configuration should be applied
        assert validator.config['enabled'] is True
        assert validator.config['performance_mode'] == 'fast'
        assert validator.thresholds['minimum_quality_score'] == 0.8


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])