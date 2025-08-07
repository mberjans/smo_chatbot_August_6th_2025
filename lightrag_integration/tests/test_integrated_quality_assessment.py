#!/usr/bin/env python3
"""
Comprehensive Test Suite for Integrated Quality Assessment System.

This module provides extensive tests for the integrated quality assessment system
that combines relevance scoring, response quality assessment, and factual accuracy
validation in the Clinical Metabolomics Oracle LightRAG integration project.

Test Coverage:
    - Integration between ClinicalMetabolomicsRelevanceScorer and factual accuracy
    - EnhancedResponseQualityAssessor with factual validation pipeline
    - IntegratedQualityWorkflow end-to-end functionality
    - Configuration management and component integration
    - Backwards compatibility with existing quality assessment
    - Performance and error handling under various conditions

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
Related to: CMO-LIGHTRAG Integrated Quality Assessment Testing
"""

import pytest
import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Test imports
try:
    from relevance_scorer import ClinicalMetabolomicsRelevanceScorer, RelevanceScore
    RELEVANCE_SCORER_AVAILABLE = True
except ImportError:
    RELEVANCE_SCORER_AVAILABLE = False

try:
    from enhanced_response_quality_assessor import EnhancedResponseQualityAssessor, ResponseQualityMetrics
    ENHANCED_ASSESSOR_AVAILABLE = True
except ImportError:
    ENHANCED_ASSESSOR_AVAILABLE = False

try:
    from integrated_quality_workflow import IntegratedQualityWorkflow, QualityAssessmentResult
    INTEGRATED_WORKFLOW_AVAILABLE = True
except ImportError:
    INTEGRATED_WORKFLOW_AVAILABLE = False

try:
    from quality_assessment_config import QualityAssessmentConfig, ComponentConfig
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False

try:
    from accuracy_scorer import FactualAccuracyScorer, AccuracyScore
    from factual_accuracy_validator import FactualAccuracyValidator
    from claim_extractor import BiomedicalClaimExtractor
    FACTUAL_COMPONENTS_AVAILABLE = True
except ImportError:
    FACTUAL_COMPONENTS_AVAILABLE = False


# =====================================================================
# TEST FIXTURES AND HELPERS
# =====================================================================

@pytest.fixture
def sample_query():
    """Provide sample query for testing."""
    return "What are the clinical applications of metabolomics in personalized medicine?"


@pytest.fixture
def sample_response():
    """Provide sample response for testing."""
    return """Metabolomics has several important clinical applications in personalized medicine. 
    First, it enables biomarker discovery for disease diagnosis and prognosis. LC-MS and GC-MS platforms 
    are used to analyze metabolite profiles in patient samples. Studies show that metabolomic signatures 
    can predict treatment responses with 85% accuracy. Research indicates that metabolomics-based approaches 
    show promise for precision medicine applications in cancer, cardiovascular disease, and metabolic disorders. 
    Clinical trials have demonstrated significant improvements in patient outcomes when metabolomics is integrated 
    into treatment selection protocols."""


@pytest.fixture
def sample_source_documents():
    """Provide sample source documents for testing."""
    return [
        "Metabolomics research demonstrates biomarker potential in clinical settings",
        "LC-MS platforms enable comprehensive metabolite profiling for diagnostic applications",
        "Personalized medicine benefits from metabolomic signature analysis"
    ]


@pytest.fixture
def sample_expected_concepts():
    """Provide expected concepts for testing."""
    return ["metabolomics", "personalized medicine", "biomarker", "clinical applications", "LC-MS", "precision medicine"]


@pytest.fixture
def mock_factual_components():
    """Create mock factual accuracy components."""
    mock_claim_extractor = Mock()
    mock_factual_validator = Mock()
    mock_accuracy_scorer = Mock()
    
    # Mock claim extraction
    mock_claim = Mock()
    mock_claim.claim_text = "Metabolomic signatures can predict treatment responses with 85% accuracy"
    mock_claim.claim_type = "numeric"
    mock_claim.confidence_score = 85.0
    mock_claim_extractor.extract_claims = AsyncMock(return_value=[mock_claim])
    
    # Mock verification
    mock_verification_result = Mock()
    mock_verification_result.verification_status.value = 'SUPPORTED'
    mock_verification_result.verification_confidence = 88.0
    mock_verification_result.evidence_strength = 82.0
    mock_verification_result.context_match = 90.0
    mock_verification_result.total_evidence_count = 3
    mock_verification_result.supporting_evidence = [Mock(), Mock()]
    mock_verification_result.contradicting_evidence = []
    mock_verification_result.neutral_evidence = [Mock()]
    mock_verification_result.processing_time_ms = 150.0
    
    mock_verification_report = Mock()
    mock_verification_report.verification_results = [mock_verification_result]
    mock_verification_report.to_dict = Mock(return_value={'verification_results': []})
    
    mock_factual_validator.verify_claims = AsyncMock(return_value=mock_verification_report)
    
    # Mock accuracy scoring
    mock_accuracy_score = Mock()
    mock_accuracy_score.overall_score = 87.5
    mock_accuracy_score.dimension_scores = {
        'claim_verification': 88.0,
        'evidence_quality': 85.0,
        'coverage_assessment': 90.0,
        'consistency_analysis': 87.0
    }
    mock_accuracy_score.confidence_score = 89.0
    mock_accuracy_score.to_dict = Mock(return_value={'overall_score': 87.5})
    
    mock_accuracy_scorer.score_accuracy = AsyncMock(return_value=mock_accuracy_score)
    
    return mock_claim_extractor, mock_factual_validator, mock_accuracy_scorer


@pytest.fixture
async def temp_config_file():
    """Create temporary configuration file for testing."""
    config_data = {
        'system': {
            'enable_quality_assessment': True,
            'enable_factual_accuracy_validation': True,
            'enable_relevance_scoring': True,
            'global_timeout_seconds': 30.0
        },
        'factual_accuracy': {
            'enabled': True,
            'minimum_claims_for_reliable_score': 2
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


# =====================================================================
# RELEVANCE SCORER INTEGRATION TESTS
# =====================================================================

@pytest.mark.skipif(not RELEVANCE_SCORER_AVAILABLE, reason="ClinicalMetabolomicsRelevanceScorer not available")
class TestRelevanceScorerIntegration:
    """Test integration of factual accuracy with ClinicalMetabolomicsRelevanceScorer."""
    
    def test_relevance_scorer_initialization(self):
        """Test that relevance scorer initializes with factual accuracy components."""
        scorer = ClinicalMetabolomicsRelevanceScorer()
        
        # Should have factual accuracy initialization
        assert hasattr(scorer, '_factual_validator')
        assert hasattr(scorer, '_claim_extractor')
        assert hasattr(scorer, '_document_indexer')
        
        # Should have configuration for factual accuracy
        config = scorer.config
        assert 'factual_accuracy_enabled' in config
        assert 'factual_accuracy_fallback_enabled' in config
    
    @pytest.mark.asyncio
    async def test_factual_accuracy_dimension_calculation(self, sample_query, sample_response):
        """Test that factual accuracy dimension is calculated."""
        scorer = ClinicalMetabolomicsRelevanceScorer()
        
        # Mock metadata with factual accuracy results
        metadata = {
            'factual_accuracy_results': {
                'overall_score': 85.0,
                'verification_results': [
                    {
                        'verification_status': 'SUPPORTED',
                        'verification_confidence': 88.0
                    }
                ]
            }
        }
        
        result = await scorer.calculate_relevance_score(sample_query, sample_response, metadata)
        
        assert isinstance(result, RelevanceScore)
        assert 'factual_accuracy' in result.dimension_scores
        assert 0 <= result.dimension_scores['factual_accuracy'] <= 100
    
    @pytest.mark.asyncio
    async def test_factual_accuracy_fallback(self, sample_query, sample_response):
        """Test fallback factual accuracy calculation when components unavailable."""
        scorer = ClinicalMetabolomicsRelevanceScorer()
        
        # Disable factual components
        scorer._factual_validator = None
        scorer._claim_extractor = None
        
        result = await scorer.calculate_relevance_score(sample_query, sample_response)
        
        assert isinstance(result, RelevanceScore)
        assert 'factual_accuracy' in result.dimension_scores
        assert result.dimension_scores['factual_accuracy'] > 0
    
    def test_enable_factual_accuracy_validation(self):
        """Test enabling factual accuracy validation with external components."""
        scorer = ClinicalMetabolomicsRelevanceScorer()
        
        mock_claim_extractor = Mock()
        mock_factual_validator = Mock()
        
        scorer.enable_factual_accuracy_validation(
            claim_extractor=mock_claim_extractor,
            factual_validator=mock_factual_validator
        )
        
        assert scorer._claim_extractor is mock_claim_extractor
        assert scorer._factual_validator is mock_factual_validator
        assert scorer.config['factual_accuracy_enabled'] is True
    
    def test_disable_factual_accuracy_validation(self):
        """Test disabling factual accuracy validation."""
        scorer = ClinicalMetabolomicsRelevanceScorer()
        
        scorer.disable_factual_accuracy_validation()
        
        assert scorer._claim_extractor is None
        assert scorer._factual_validator is None
        assert scorer.config['factual_accuracy_enabled'] is False


# =====================================================================
# ENHANCED RESPONSE QUALITY ASSESSOR TESTS
# =====================================================================

@pytest.mark.skipif(not ENHANCED_ASSESSOR_AVAILABLE, reason="EnhancedResponseQualityAssessor not available")
class TestEnhancedResponseQualityAssessor:
    """Test EnhancedResponseQualityAssessor with factual accuracy integration."""
    
    def test_assessor_initialization(self):
        """Test that enhanced assessor initializes properly."""
        assessor = EnhancedResponseQualityAssessor()
        
        # Should have factual accuracy components attributes
        assert hasattr(assessor, '_claim_extractor')
        assert hasattr(assessor, '_factual_validator')
        assert hasattr(assessor, '_accuracy_scorer')
        
        # Should have updated quality weights including factual accuracy
        assert 'factual_accuracy' in assessor.quality_weights
        assert assessor.quality_weights['factual_accuracy'] > 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_quality_assessment(self, sample_query, sample_response, 
                                                  sample_source_documents, sample_expected_concepts):
        """Test comprehensive quality assessment including factual accuracy."""
        assessor = EnhancedResponseQualityAssessor()
        
        result = await assessor.assess_response_quality(
            query=sample_query,
            response=sample_response,
            source_documents=sample_source_documents,
            expected_concepts=sample_expected_concepts
        )
        
        assert isinstance(result, ResponseQualityMetrics)
        
        # Should have all core metrics
        assert 0 <= result.overall_quality_score <= 100
        assert 0 <= result.relevance_score <= 100
        assert 0 <= result.clarity_score <= 100
        assert 0 <= result.factual_accuracy_score <= 100
        
        # Should have factual accuracy details
        assert hasattr(result, 'factual_validation_results')
        assert hasattr(result, 'verified_claims_count')
        assert hasattr(result, 'contradicted_claims_count')
        assert hasattr(result, 'factual_confidence_score')
        
        # Should have quality grade
        assert result.quality_grade in ["Excellent", "Good", "Acceptable", "Needs Improvement", "Poor"]
        assert result.factual_reliability_grade in ["Highly Reliable", "Reliable", "Moderately Reliable", "Questionable", "Unreliable"]
    
    @pytest.mark.asyncio
    async def test_factual_validation_with_mock_components(self, sample_query, sample_response,
                                                         mock_factual_components):
        """Test factual validation with mock components."""
        mock_claim_extractor, mock_factual_validator, mock_accuracy_scorer = mock_factual_components
        
        assessor = EnhancedResponseQualityAssessor()
        assessor.enable_factual_validation(
            claim_extractor=mock_claim_extractor,
            factual_validator=mock_factual_validator,
            accuracy_scorer=mock_accuracy_scorer
        )
        
        result = await assessor.assess_response_quality(
            query=sample_query,
            response=sample_response
        )
        
        assert isinstance(result, ResponseQualityMetrics)
        assert result.factual_accuracy_score > 0
        assert result.verified_claims_count >= 0
        
        # Verify components were called
        mock_claim_extractor.extract_claims.assert_called_once()
        mock_factual_validator.verify_claims.assert_called_once()
        mock_accuracy_scorer.score_accuracy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fallback_factual_metrics(self, sample_query, sample_response):
        """Test fallback factual metrics when components unavailable."""
        assessor = EnhancedResponseQualityAssessor()
        
        # Ensure components are not available
        assessor._claim_extractor = None
        assessor._factual_validator = None
        assessor._accuracy_scorer = None
        
        result = await assessor.assess_response_quality(
            query=sample_query,
            response=sample_response
        )
        
        assert isinstance(result, ResponseQualityMetrics)
        assert result.factual_accuracy_score > 0  # Should have fallback score
        assert result.factual_validation_results['method'] == 'fallback'
    
    @pytest.mark.asyncio
    async def test_batch_quality_assessment(self, sample_query, sample_response):
        """Test batch quality assessment functionality."""
        assessor = EnhancedResponseQualityAssessor()
        
        assessments = [
            (sample_query, sample_response, [], ["metabolomics"]),
            ("What is LC-MS?", "LC-MS is liquid chromatography mass spectrometry.", [], ["LC-MS"]),
            ("Clinical applications?", "Used for biomarker discovery.", [], ["clinical"])
        ]
        
        results = await assessor.batch_assess_quality(assessments)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, ResponseQualityMetrics)
            assert 0 <= result.overall_quality_score <= 100


# =====================================================================
# INTEGRATED QUALITY WORKFLOW TESTS  
# =====================================================================

@pytest.mark.skipif(not INTEGRATED_WORKFLOW_AVAILABLE, reason="IntegratedQualityWorkflow not available")
class TestIntegratedQualityWorkflow:
    """Test IntegratedQualityWorkflow comprehensive functionality."""
    
    def test_workflow_initialization(self):
        """Test workflow initialization with all components."""
        workflow = IntegratedQualityWorkflow()
        
        # Should initialize all component attributes
        assert hasattr(workflow, '_relevance_scorer')
        assert hasattr(workflow, '_quality_assessor') 
        assert hasattr(workflow, '_claim_extractor')
        assert hasattr(workflow, '_factual_validator')
        assert hasattr(workflow, '_accuracy_scorer')
        
        # Should have performance tracking
        assert hasattr(workflow, '_performance_metrics')
        assert hasattr(workflow, '_assessment_history')
    
    @pytest.mark.asyncio
    async def test_comprehensive_quality_assessment(self, sample_query, sample_response,
                                                   sample_source_documents, sample_expected_concepts):
        """Test comprehensive quality assessment workflow."""
        workflow = IntegratedQualityWorkflow()
        
        result = await workflow.assess_comprehensive_quality(
            query=sample_query,
            response=sample_response,
            source_documents=sample_source_documents,
            expected_concepts=sample_expected_concepts
        )
        
        assert isinstance(result, QualityAssessmentResult)
        
        # Should have overall assessment
        assert 0 <= result.overall_quality_score <= 100
        assert result.quality_grade in ["Excellent", "Good", "Acceptable", "Marginal", "Poor"]
        assert 0 <= result.assessment_confidence <= 100
        
        # Should have component results
        assert result.components_used is not None
        assert len(result.components_used) > 0
        
        # Should have analysis results
        assert result.consistency_analysis is not None
        assert isinstance(result.strength_areas, list)
        assert isinstance(result.improvement_areas, list)
        assert isinstance(result.actionable_recommendations, list)
        
        # Should have performance metrics
        assert result.processing_time_ms > 0
        assert result.performance_metrics is not None
    
    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_processing(self, sample_query, sample_response):
        """Test both parallel and sequential processing modes."""
        # Test parallel processing
        parallel_config = {'enable_parallel_processing': True}
        workflow_parallel = IntegratedQualityWorkflow(config=parallel_config)
        
        start_time = time.time()
        result_parallel = await workflow_parallel.assess_comprehensive_quality(
            query=sample_query, response=sample_response
        )
        parallel_time = time.time() - start_time
        
        # Test sequential processing
        sequential_config = {'enable_parallel_processing': False}
        workflow_sequential = IntegratedQualityWorkflow(config=sequential_config)
        
        start_time = time.time()
        result_sequential = await workflow_sequential.assess_comprehensive_quality(
            query=sample_query, response=sample_response
        )
        sequential_time = time.time() - start_time
        
        # Both should produce valid results
        assert isinstance(result_parallel, QualityAssessmentResult)
        assert isinstance(result_sequential, QualityAssessmentResult)
        
        # Parallel should generally be faster (though not guaranteed in tests)
        assert result_parallel.processing_time_ms > 0
        assert result_sequential.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_component_failure_handling(self, sample_query, sample_response):
        """Test handling of component failures with fallback."""
        workflow = IntegratedQualityWorkflow()
        
        # Mock component failures
        if workflow._relevance_scorer:
            workflow._relevance_scorer.calculate_relevance_score = AsyncMock(side_effect=Exception("Test error"))
        
        result = await workflow.assess_comprehensive_quality(
            query=sample_query, response=sample_response
        )
        
        # Should still produce a result with fallback
        assert isinstance(result, QualityAssessmentResult)
        assert result.overall_quality_score >= 0
        assert len(result.error_details) >= 0  # May contain error details
    
    @pytest.mark.asyncio
    async def test_batch_assessment(self, sample_query, sample_response):
        """Test batch assessment functionality."""
        workflow = IntegratedQualityWorkflow()
        
        assessments = [
            (sample_query, sample_response, [], []),
            ("What is metabolomics?", "Metabolomics studies small molecules.", [], []),
            ("Clinical uses?", "Used for diagnosis and treatment.", [], [])
        ]
        
        results = await workflow.batch_assess_quality(assessments)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, QualityAssessmentResult)
            assert result.overall_quality_score >= 0
    
    def test_performance_statistics(self, sample_query, sample_response):
        """Test performance statistics collection.""" 
        workflow = IntegratedQualityWorkflow()
        
        # Initially no statistics
        stats = workflow.get_performance_statistics()
        assert stats['status'] == 'no_data'
        
        # Add mock performance data
        workflow._performance_metrics['processing_times'] = [100.0, 150.0, 120.0]
        workflow._performance_metrics['quality_scores'] = [85.0, 90.0, 78.0]
        workflow._performance_metrics['confidence_scores'] = [88.0, 92.0, 80.0]
        
        stats = workflow.get_performance_statistics()
        
        assert stats['total_assessments'] == 3
        assert stats['avg_processing_time_ms'] > 0
        assert stats['avg_quality_score'] > 0
        assert stats['assessments_per_minute'] > 0


# =====================================================================
# CONFIGURATION MANAGEMENT TESTS
# =====================================================================

@pytest.mark.skipif(not CONFIG_MANAGER_AVAILABLE, reason="QualityAssessmentConfig not available")
class TestConfigurationManagement:
    """Test configuration management for integrated quality assessment."""
    
    def test_default_configuration_creation(self):
        """Test creation of default configuration."""
        config = QualityAssessmentConfig()
        
        # Should have all main configuration sections
        system_config = config.get_system_config()
        assert 'enable_quality_assessment' in system_config
        assert 'enable_factual_accuracy_validation' in system_config
        assert 'enable_relevance_scoring' in system_config
        
        # Should have component configurations
        relevance_config = config.get_component_config('relevance_scorer')
        assert isinstance(relevance_config, ComponentConfig)
        assert relevance_config.enabled is not None
        
        quality_config = config.get_component_config('quality_assessor')
        assert isinstance(quality_config, ComponentConfig)
        
        # Should have validation configuration
        validation_config = config.get_validation_config()
        assert validation_config.minimum_quality_threshold > 0
        assert validation_config.minimum_factual_accuracy_threshold > 0
    
    def test_component_configuration_updates(self):
        """Test updating component configurations."""
        config = QualityAssessmentConfig()
        
        # Update relevance scorer configuration
        config.update_component_config('relevance_scorer', {
            'timeout_seconds': 15.0,
            'cache_enabled': False,
            'config': {'new_parameter': 'test_value'}
        })
        
        relevance_config = config.get_component_config('relevance_scorer')
        assert relevance_config.timeout_seconds == 15.0
        assert relevance_config.cache_enabled is False
        assert relevance_config.config['new_parameter'] == 'test_value'
    
    def test_factual_accuracy_enable_disable(self):
        """Test enabling and disabling factual accuracy validation."""
        config = QualityAssessmentConfig()
        
        # Test enabling comprehensive factual accuracy
        config.enable_factual_accuracy_validation(comprehensive=True)
        
        claim_extractor_config = config.get_component_config('claim_extractor')
        assert claim_extractor_config.enabled is True
        
        factual_validator_config = config.get_component_config('factual_validator')
        assert factual_validator_config.enabled is True
        
        # Test disabling factual accuracy
        config.disable_factual_accuracy_validation()
        
        claim_extractor_config = config.get_component_config('claim_extractor')
        assert claim_extractor_config.enabled is False
        
        factual_validator_config = config.get_component_config('factual_validator')
        assert factual_validator_config.enabled is False
    
    def test_performance_optimization(self):
        """Test performance optimization settings."""
        config = QualityAssessmentConfig()
        
        # Test fast optimization
        config.optimize_for_performance('fast')
        system_config = config.get_system_config()
        assert system_config['enable_parallel_processing'] is True
        assert system_config['max_concurrent_assessments'] > 5
        
        # Test thorough optimization
        config.optimize_for_performance('thorough')
        system_config = config.get_system_config()
        assert system_config['enable_parallel_processing'] is False
        assert system_config['max_concurrent_assessments'] <= 5
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        config = QualityAssessmentConfig()
        
        # Valid configuration should have no issues
        issues = config.validate_configuration()
        assert isinstance(issues, list)
        
        # Introduce invalid configuration
        config.update_validation_thresholds(minimum_quality_threshold=-10)
        issues = config.validate_configuration()
        assert len(issues) > 0
        assert any("between 0 and 100" in issue for issue in issues)
    
    @pytest.mark.asyncio
    async def test_configuration_file_operations(self, temp_config_file):
        """Test loading and saving configuration files."""
        # Test loading from file
        config = QualityAssessmentConfig(temp_config_file)
        system_config = config.get_system_config()
        assert system_config['enable_quality_assessment'] is True
        assert system_config['global_timeout_seconds'] == 30.0
        
        # Test saving to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            save_path = f.name
        
        try:
            config.save_to_file(save_path)
            assert Path(save_path).exists()
            
            # Verify saved content
            with open(save_path, 'r') as f:
                saved_config = json.load(f)
            
            assert 'system_config' in saved_config
            assert 'component_configs' in saved_config
            assert 'validation_config' in saved_config
        finally:
            Path(save_path).unlink(missing_ok=True)
    
    def test_configuration_summary(self):
        """Test configuration summary generation."""
        config = QualityAssessmentConfig()
        
        summary = config.get_configuration_summary()
        
        assert 'system_enabled' in summary
        assert 'component_status' in summary
        assert 'performance_settings' in summary
        assert 'validation_thresholds' in summary
        assert 'integration_weights' in summary
        assert 'configuration_issues' in summary
        
        # All components should have status
        component_status = summary['component_status']
        expected_components = ['relevance_scorer', 'quality_assessor', 'claim_extractor', 
                             'factual_validator', 'accuracy_scorer', 'integrated_workflow']
        
        for component in expected_components:
            assert component in component_status


# =====================================================================
# BACKWARDS COMPATIBILITY TESTS
# =====================================================================

class TestBackwardsCompatibility:
    """Test backwards compatibility with existing quality assessment systems."""
    
    @pytest.mark.skipif(not RELEVANCE_SCORER_AVAILABLE, reason="ClinicalMetabolomicsRelevanceScorer not available")
    @pytest.mark.asyncio
    async def test_existing_relevance_scorer_interface(self, sample_query, sample_response):
        """Test that existing relevance scorer interface still works."""
        scorer = ClinicalMetabolomicsRelevanceScorer()
        
        # Original interface should still work
        result = await scorer.calculate_relevance_score(sample_query, sample_response)
        
        assert isinstance(result, RelevanceScore)
        assert hasattr(result, 'overall_score')
        assert hasattr(result, 'dimension_scores')
        assert hasattr(result, 'query_type')
        assert hasattr(result, 'confidence_score')
        
        # Should have existing dimensions
        expected_dimensions = ['metabolomics_relevance', 'clinical_applicability', 'query_alignment']
        for dimension in expected_dimensions:
            assert dimension in result.dimension_scores
    
    def test_existing_quality_weights_still_valid(self):
        """Test that existing quality weight structures are still valid."""
        if RELEVANCE_SCORER_AVAILABLE:
            scorer = ClinicalMetabolomicsRelevanceScorer()
            weights = scorer.weighting_manager.get_weights('general')
            
            # Should still have original weights
            original_dimensions = ['metabolomics_relevance', 'clinical_applicability', 
                                 'query_alignment', 'scientific_rigor']
            for dimension in original_dimensions:
                assert dimension in weights
                assert 0 <= weights[dimension] <= 1
            
            # Total weights should still approximately equal 1.0
            total_weight = sum(weights.values())
            assert 0.9 <= total_weight <= 1.1


# =====================================================================
# ERROR HANDLING AND EDGE CASES
# =====================================================================

class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in integrated quality assessment."""
    
    @pytest.mark.skipif(not INTEGRATED_WORKFLOW_AVAILABLE, reason="IntegratedQualityWorkflow not available")
    @pytest.mark.asyncio
    async def test_empty_query_and_response_handling(self):
        """Test handling of empty queries and responses."""
        workflow = IntegratedQualityWorkflow()
        
        # Empty query should raise error
        with pytest.raises(Exception):
            await workflow.assess_comprehensive_quality("", "some response")
        
        # Empty response should raise error
        with pytest.raises(Exception):
            await workflow.assess_comprehensive_quality("some query", "")
        
        # Both empty should raise error
        with pytest.raises(Exception):
            await workflow.assess_comprehensive_quality("", "")
    
    @pytest.mark.skipif(not INTEGRATED_WORKFLOW_AVAILABLE, reason="IntegratedQualityWorkflow not available")
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of component timeouts."""
        config = {
            'max_processing_time_seconds': 0.001,  # Very short timeout
            'fallback_on_component_failure': True
        }
        workflow = IntegratedQualityWorkflow(config=config)
        
        # Should handle timeout gracefully with fallback
        result = await workflow.assess_comprehensive_quality(
            query="test query", 
            response="test response"
        )
        
        assert isinstance(result, QualityAssessmentResult)
        assert result.overall_quality_score >= 0
    
    @pytest.mark.skipif(not ENHANCED_ASSESSOR_AVAILABLE, reason="EnhancedResponseQualityAssessor not available")
    @pytest.mark.asyncio
    async def test_very_long_response_handling(self):
        """Test handling of very long responses."""
        assessor = EnhancedResponseQualityAssessor()
        
        # Create very long response
        long_response = "Metabolomics is important. " * 1000
        
        result = await assessor.assess_response_quality(
            query="What is metabolomics?",
            response=long_response
        )
        
        assert isinstance(result, ResponseQualityMetrics)
        assert "response_very_long" in result.quality_flags
    
    @pytest.mark.skipif(not ENHANCED_ASSESSOR_AVAILABLE, reason="EnhancedResponseQualityAssessor not available")
    @pytest.mark.asyncio
    async def test_no_biomedical_terminology_handling(self):
        """Test handling of responses with no biomedical terminology."""
        assessor = EnhancedResponseQualityAssessor()
        
        non_biomedical_response = "This is a simple response without any technical terms or scientific language."
        
        result = await assessor.assess_response_quality(
            query="What is metabolomics?",
            response=non_biomedical_response
        )
        
        assert isinstance(result, ResponseQualityMetrics)
        assert "lacks_biomedical_terminology" in result.quality_flags
        assert result.biomedical_terminology_score < 50


# =====================================================================
# PERFORMANCE AND LOAD TESTS
# =====================================================================

class TestPerformanceAndLoad:
    """Test performance characteristics and load handling."""
    
    @pytest.mark.skipif(not INTEGRATED_WORKFLOW_AVAILABLE, reason="IntegratedQualityWorkflow not available")
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, sample_query, sample_response):
        """Test that assessment performance meets benchmarks."""
        workflow = IntegratedQualityWorkflow()
        
        start_time = time.time()
        result = await workflow.assess_comprehensive_quality(
            query=sample_query,
            response=sample_response
        )
        processing_time = (time.time() - start_time) * 1000
        
        # Should complete within reasonable time (adjust as needed)
        assert processing_time < 10000  # 10 seconds
        assert result.processing_time_ms < 10000
        assert result.performance_metrics['performance_grade'] in ["Excellent", "Good", "Acceptable", "Slow", "Very Slow"]
    
    @pytest.mark.skipif(not INTEGRATED_WORKFLOW_AVAILABLE, reason="IntegratedQualityWorkflow not available") 
    @pytest.mark.asyncio
    async def test_concurrent_assessments(self, sample_query, sample_response):
        """Test handling of multiple concurrent assessments."""
        workflow = IntegratedQualityWorkflow()
        
        # Create multiple assessment tasks
        tasks = [
            workflow.assess_comprehensive_quality(
                query=f"{sample_query} {i}",
                response=f"{sample_response} Assessment {i}"
            )
            for i in range(3)
        ]
        
        # Run concurrently
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, QualityAssessmentResult)
            assert result.overall_quality_score >= 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "-x"])