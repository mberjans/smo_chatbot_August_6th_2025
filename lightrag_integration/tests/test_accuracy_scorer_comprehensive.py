#!/usr/bin/env python3
"""
Comprehensive Test Suite for Factual Accuracy Scoring System.

This test suite provides thorough testing for the FactualAccuracyScorer class
including all scoring dimensions, integration capabilities, and error handling.

Test Categories:
1. Unit tests for individual scoring methods
2. Integration tests with verification results
3. Performance and scalability tests
4. Configuration and customization tests
5. Error handling and edge case tests
6. Report generation tests
7. Quality integration tests

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import asyncio
import json
import time
import tempfile
import statistics
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import asdict

# Import test fixtures
from .factual_validation_test_fixtures import *

# Import the modules to test
try:
    from ..accuracy_scorer import (
        FactualAccuracyScorer, AccuracyScore, AccuracyReport, AccuracyMetrics,
        AccuracyGrade, AccuracyScoringError, ReportGenerationError, QualityIntegrationError,
        score_verification_results, generate_accuracy_report, integrate_quality_assessment
    )
    from ..factual_accuracy_validator import (
        VerificationResult, VerificationStatus, EvidenceItem
    )
    from ..claim_extractor import ExtractedClaim
    from ..relevance_scorer import ClinicalMetabolomicsRelevanceScorer, RelevanceScore
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


@pytest.mark.accuracy_scorer
class TestFactualAccuracyScorer:
    """Comprehensive test suite for FactualAccuracyScorer core functionality."""
    
    @pytest.fixture
    def scorer_config(self):
        """Provide test configuration for scorer."""
        return {
            'scoring_weights': {
                'claim_verification': 0.40,
                'evidence_quality': 0.25,
                'coverage_assessment': 0.20,
                'consistency_analysis': 0.10,
                'confidence_factor': 0.05
            },
            'claim_type_weights': {
                'numeric': 1.2,
                'qualitative': 1.0,
                'methodological': 1.1,
                'temporal': 0.9,
                'comparative': 1.1,
                'general': 0.8
            },
            'evidence_quality_thresholds': {
                'high_quality': 85.0,
                'medium_quality': 65.0,
                'low_quality': 45.0
            },
            'performance_targets': {
                'max_processing_time_ms': 3000,
                'min_claims_for_reliable_score': 3,
                'max_error_rate': 0.03
            },
            'integration_settings': {
                'enable_relevance_integration': True,
                'quality_system_compatibility': True,
                'generate_integration_data': True
            }
        }
    
    @pytest.fixture
    def scorer(self, scorer_config):
        """Create FactualAccuracyScorer instance for testing."""
        return FactualAccuracyScorer(config=scorer_config)
    
    @pytest.fixture 
    def mock_relevance_scorer(self):
        """Create mock relevance scorer for integration testing."""
        relevance_scorer = Mock(spec=ClinicalMetabolomicsRelevanceScorer)
        
        mock_relevance_score = Mock(spec=RelevanceScore)
        mock_relevance_score.overall_score = 78.5
        mock_relevance_score.relevance_grade = "Good"
        mock_relevance_score.dimension_scores = {
            "scientific_rigor": 82.0,
            "biomedical_context_depth": 75.0,
            "query_alignment": 80.0,
            "metabolomics_relevance": 77.0,
            "clinical_applicability": 76.0
        }
        mock_relevance_score.query_type = "biomedical"
        mock_relevance_score.confidence_score = 79.0
        
        relevance_scorer.calculate_relevance_score = AsyncMock(return_value=mock_relevance_score)
        return relevance_scorer
    
    # Initialization and Configuration Tests
    
    @pytest.mark.asyncio
    async def test_scorer_initialization(self, scorer_config):
        """Test scorer initialization with custom configuration."""
        
        scorer = FactualAccuracyScorer(config=scorer_config)
        
        assert scorer.config == scorer_config
        assert scorer.scoring_weights == scorer_config['scoring_weights']
        assert scorer.claim_type_weights == scorer_config['claim_type_weights']
        assert len(scorer.grading_thresholds) == 5
        assert hasattr(scorer, 'integration_parameters')
        assert scorer.scoring_stats['total_scorings'] == 0
    
    @pytest.mark.asyncio
    async def test_default_configuration(self):
        """Test scorer with default configuration."""
        
        scorer = FactualAccuracyScorer()
        
        assert scorer.config is not None
        assert 'scoring_weights' in scorer.config
        assert 'claim_type_weights' in scorer.config
        assert sum(scorer.scoring_weights.values()) == pytest.approx(1.0)
        assert all(weight > 0 for weight in scorer.scoring_weights.values())
    
    @pytest.mark.asyncio
    async def test_scorer_with_relevance_integration(self, scorer_config, mock_relevance_scorer):
        """Test scorer initialization with relevance scorer integration."""
        
        scorer = FactualAccuracyScorer(
            relevance_scorer=mock_relevance_scorer,
            config=scorer_config
        )
        
        assert scorer.relevance_scorer is mock_relevance_scorer
        assert scorer.config['integration_settings']['enable_relevance_integration']
        assert hasattr(scorer, 'dimension_mappings')
    
    # Core Scoring Tests
    
    @pytest.mark.asyncio
    async def test_score_accuracy_basic(self, scorer, sample_verification_results):
        """Test basic accuracy scoring functionality."""
        
        accuracy_score = await scorer.score_accuracy(sample_verification_results)
        
        assert isinstance(accuracy_score, AccuracyScore)
        assert 0 <= accuracy_score.overall_score <= 100
        assert isinstance(accuracy_score.grade, AccuracyGrade)
        assert accuracy_score.total_claims_assessed == len(sample_verification_results)
        assert accuracy_score.processing_time_ms > 0
        assert len(accuracy_score.dimension_scores) == 5
        assert len(accuracy_score.claim_type_scores) > 0
    
    @pytest.mark.asyncio
    async def test_score_accuracy_with_claims_context(self, scorer, sample_verification_results, sample_extracted_claims):
        """Test accuracy scoring with claims context."""
        
        accuracy_score = await scorer.score_accuracy(
            sample_verification_results, 
            claims=sample_extracted_claims
        )
        
        assert accuracy_score.metadata['has_claims_context'] is True
        assert accuracy_score.total_claims_assessed == len(sample_verification_results)
        assert 'numeric' in accuracy_score.claim_type_scores
        assert 'qualitative' in accuracy_score.claim_type_scores
    
    @pytest.mark.asyncio
    async def test_score_accuracy_empty_results(self, scorer):
        """Test scoring with empty verification results."""
        
        accuracy_score = await scorer.score_accuracy([])
        
        assert accuracy_score.overall_score == 0.0
        assert accuracy_score.grade == AccuracyGrade.POOR
        assert accuracy_score.total_claims_assessed == 0
        assert len(accuracy_score.dimension_scores) == 0
        assert len(accuracy_score.claim_type_scores) == 0
    
    @pytest.mark.asyncio
    async def test_dimension_score_calculation(self, scorer, sample_verification_results):
        """Test individual dimension score calculations."""
        
        # Test claim verification dimension
        claim_score = await scorer._calculate_claim_verification_score(sample_verification_results)
        assert 0 <= claim_score <= 100
        
        # Test evidence quality dimension
        evidence_score = await scorer._calculate_evidence_quality_score(sample_verification_results)
        assert 0 <= evidence_score <= 100
        
        # Test coverage dimension
        coverage_score = await scorer._calculate_coverage_score(sample_verification_results)
        assert 0 <= coverage_score <= 100
        
        # Test consistency dimension
        consistency_score = await scorer._calculate_consistency_score(sample_verification_results)
        assert 0 <= consistency_score <= 100
        
        # Test confidence factor
        confidence_factor = await scorer._calculate_confidence_factor(sample_verification_results)
        assert 0 <= confidence_factor <= 100
    
    @pytest.mark.asyncio
    async def test_claim_type_scoring(self, scorer, sample_verification_results):
        """Test claim type-specific scoring."""
        
        claim_type_scores = await scorer._calculate_claim_type_scores(sample_verification_results)
        
        assert isinstance(claim_type_scores, dict)
        assert len(claim_type_scores) > 0
        assert all(0 <= score <= 100 for score in claim_type_scores.values())
        
        # Test individual claim type scoring methods
        numeric_results = [r for r in sample_verification_results 
                          if r.metadata.get('claim_type') == 'numeric']
        if numeric_results:
            numeric_score = await scorer._score_numeric_claims(numeric_results)
            assert 0 <= numeric_score <= 100
    
    @pytest.mark.asyncio
    async def test_overall_score_calculation(self, scorer, sample_verification_results):
        """Test overall score calculation logic."""
        
        dimension_scores = {
            'claim_verification': 85.0,
            'evidence_quality': 78.0,
            'coverage_assessment': 82.0,
            'consistency_analysis': 79.0,
            'confidence_factor': 75.0
        }
        
        claim_type_scores = {
            'numeric': 87.0,
            'qualitative': 76.0,
            'methodological': 83.0
        }
        
        overall_score = await scorer._calculate_overall_score(
            dimension_scores, claim_type_scores, sample_verification_results
        )
        
        assert 0 <= overall_score <= 100
        assert isinstance(overall_score, float)
        
        # Test minimum claims penalty
        few_results = sample_verification_results[:1]  # Only 1 claim
        penalized_score = await scorer._calculate_overall_score(
            dimension_scores, claim_type_scores, few_results
        )
        assert penalized_score < overall_score  # Should be penalized
    
    @pytest.mark.asyncio
    async def test_accuracy_grading(self, scorer):
        """Test accuracy grade determination."""
        
        # Test different score ranges
        test_cases = [
            (95.0, AccuracyGrade.EXCELLENT),
            (85.0, AccuracyGrade.GOOD),
            (75.0, AccuracyGrade.ACCEPTABLE),
            (65.0, AccuracyGrade.MARGINAL),
            (45.0, AccuracyGrade.POOR)
        ]
        
        for score, expected_grade in test_cases:
            grade = scorer._determine_accuracy_grade(score)
            assert grade == expected_grade
    
    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self, scorer, sample_verification_results):
        """Test confidence score calculation."""
        
        dimension_scores = {
            'claim_verification': 85.0,
            'evidence_quality': 80.0,
            'coverage_assessment': 83.0,
            'consistency_analysis': 78.0,
            'confidence_factor': 82.0
        }
        
        confidence_score = await scorer._calculate_confidence_score(
            sample_verification_results, dimension_scores
        )
        
        assert 0 <= confidence_score <= 100
        assert isinstance(confidence_score, float)
    
    # Report Generation Tests
    
    @pytest.mark.asyncio
    async def test_generate_comprehensive_report(self, scorer, sample_verification_results, sample_extracted_claims):
        """Test comprehensive report generation."""
        
        test_query = "What are glucose levels in diabetes?"
        test_response = "Glucose levels were 150 mg/dL in diabetic patients."
        
        report = await scorer.generate_comprehensive_report(
            sample_verification_results,
            claims=sample_extracted_claims,
            query=test_query,
            response=test_response
        )
        
        assert isinstance(report, AccuracyReport)
        assert report.report_id.startswith("FACR_")
        assert isinstance(report.accuracy_score, AccuracyScore)
        assert isinstance(report.performance_metrics, AccuracyMetrics)
        assert len(report.quality_recommendations) > 0
        assert len(report.claims_analysis) == len(sample_verification_results)
        assert 'total_evidence_items' in report.evidence_analysis
        assert 'overall_coverage_rate' in report.coverage_analysis
    
    @pytest.mark.asyncio
    async def test_report_detailed_breakdown(self, scorer, sample_verification_results):
        """Test detailed breakdown generation in reports."""
        
        report = await scorer.generate_comprehensive_report(sample_verification_results)
        breakdown = report.detailed_breakdown
        
        assert 'status_distribution' in breakdown
        assert 'evidence_statistics' in breakdown
        assert 'confidence_distribution' in breakdown
        assert 'processing_statistics' in breakdown
        assert 'dimension_breakdown' in breakdown
        assert 'claim_type_breakdown' in breakdown
    
    @pytest.mark.asyncio
    async def test_summary_statistics_generation(self, scorer, sample_verification_results):
        """Test summary statistics generation."""
        
        summary = await scorer._generate_summary_statistics(
            sample_verification_results, 
            sample_accuracy_score
        )
        
        required_fields = [
            'total_claims', 'verified_claims', 'verification_rate',
            'support_rate', 'contradiction_rate', 'high_confidence_rate',
            'high_evidence_rate', 'overall_accuracy_score', 'accuracy_grade',
            'reliability_indicator'
        ]
        
        for field in required_fields:
            assert field in summary
        
        assert 0 <= summary['verification_rate'] <= 1
        assert 0 <= summary['support_rate'] <= 1
        assert summary['total_claims'] == len(sample_verification_results)
    
    @pytest.mark.asyncio
    async def test_performance_metrics_generation(self, scorer, sample_verification_results):
        """Test performance metrics generation."""
        
        start_time = time.time()
        metrics = await scorer._generate_performance_metrics(sample_verification_results, start_time)
        
        assert isinstance(metrics, AccuracyMetrics)
        assert 'total_verification_time_ms' in metrics.verification_performance
        assert 'total_scoring_time_ms' in metrics.scoring_performance
        assert 'error_rate' in metrics.quality_indicators
        assert 'memory_efficient' in metrics.system_health
    
    @pytest.mark.asyncio
    async def test_quality_recommendations_generation(self, scorer, sample_verification_results):
        """Test quality recommendations generation."""
        
        # Test with poor accuracy score
        poor_score = Mock(spec=AccuracyScore)
        poor_score.overall_score = 45.0
        poor_score.evidence_quality_score = 40.0
        poor_score.coverage_score = 35.0
        poor_score.consistency_score = 42.0
        poor_score.confidence_score = 38.0
        poor_score.claim_type_scores = {'numeric': 45.0, 'qualitative': 40.0}
        
        recommendations = await scorer._generate_quality_recommendations(
            poor_score, sample_verification_results, None
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('low' in rec.lower() for rec in recommendations)
        
        # Test with good accuracy score
        good_score = Mock(spec=AccuracyScore)
        good_score.overall_score = 88.0
        good_score.evidence_quality_score = 85.0
        good_score.coverage_score = 87.0
        good_score.consistency_score = 86.0
        good_score.confidence_score = 84.0
        good_score.claim_type_scores = {'numeric': 89.0, 'qualitative': 85.0}
        good_score.total_claims_assessed = 10
        
        good_recommendations = await scorer._generate_quality_recommendations(
            good_score, sample_verification_results, None
        )
        
        assert len(good_recommendations) > 0
        assert any('maintain' in rec.lower() or 'performing well' in rec.lower() 
                  for rec in good_recommendations)
    
    @pytest.mark.asyncio
    async def test_claims_analysis_generation(self, scorer, sample_verification_results, sample_extracted_claims):
        """Test individual claims analysis generation."""
        
        analysis = await scorer._generate_claims_analysis(sample_verification_results, sample_extracted_claims)
        
        assert isinstance(analysis, list)
        assert len(analysis) == len(sample_verification_results)
        
        for claim_analysis in analysis:
            required_fields = [
                'claim_id', 'verification_status', 'verification_confidence',
                'evidence_strength', 'context_match', 'processing_time_ms',
                'evidence_summary', 'verification_strategy'
            ]
            
            for field in required_fields:
                assert field in claim_analysis
            
            assert 'supporting_count' in claim_analysis['evidence_summary']
            assert 'contradicting_count' in claim_analysis['evidence_summary']
            assert 'total_evidence' in claim_analysis['evidence_summary']
    
    @pytest.mark.asyncio
    async def test_evidence_analysis_generation(self, scorer, sample_verification_results):
        """Test evidence analysis generation."""
        
        analysis = await scorer._generate_evidence_analysis(sample_verification_results)
        
        if analysis.get('total_evidence_items', 0) > 0:
            required_fields = [
                'total_evidence_items', 'unique_sources', 'source_distribution',
                'evidence_type_distribution', 'quality_distribution',
                'confidence_statistics', 'average_evidence_per_claim', 'top_sources'
            ]
            
            for field in required_fields:
                assert field in analysis
            
            assert 'mean' in analysis['confidence_statistics']
            assert 'high_quality' in analysis['quality_distribution']
        else:
            assert analysis['total_evidence_items'] == 0
            assert 'message' in analysis
    
    @pytest.mark.asyncio
    async def test_coverage_analysis_generation(self, scorer, sample_verification_results, sample_extracted_claims):
        """Test coverage analysis generation."""
        
        analysis = await scorer._generate_coverage_analysis(sample_verification_results, sample_extracted_claims)
        
        required_fields = [
            'total_claims', 'claims_with_evidence', 'claims_with_support',
            'claims_without_evidence', 'overall_coverage_rate', 'support_coverage_rate',
            'coverage_by_claim_type', 'sources_utilized', 'coverage_quality'
        ]
        
        for field in required_fields:
            assert field in analysis
        
        assert 0 <= analysis['overall_coverage_rate'] <= 1
        assert 0 <= analysis['support_coverage_rate'] <= 1
        assert analysis['total_claims'] == len(sample_verification_results)
        assert 'recommendations' in analysis
    
    # Integration Tests
    
    @pytest.mark.asyncio
    async def test_relevance_scorer_integration(self, scorer_config, mock_relevance_scorer, sample_accuracy_score):
        """Test integration with relevance scorer."""
        
        scorer = FactualAccuracyScorer(
            relevance_scorer=mock_relevance_scorer,
            config=scorer_config
        )
        
        test_query = "What are glucose levels in diabetes?"
        test_response = "Glucose levels were 150 mg/dL in diabetic patients."
        
        integration_result = await scorer.integrate_with_relevance_scorer(
            sample_accuracy_score, test_query, test_response
        )
        
        assert isinstance(integration_result, dict)
        assert 'factual_accuracy' in integration_result
        assert 'relevance_assessment' in integration_result
        assert 'integrated_quality' in integration_result
        assert 'integration_metadata' in integration_result
        
        # Check integrated quality metrics
        integrated_quality = integration_result['integrated_quality']
        assert 'combined_score' in integrated_quality
        assert 'quality_grade' in integrated_quality
        assert 'strength_areas' in integrated_quality
        assert 'improvement_areas' in integrated_quality
        assert 'overall_assessment' in integrated_quality
        
        assert 0 <= integrated_quality['combined_score'] <= 100
        assert isinstance(integrated_quality['strength_areas'], list)
        assert isinstance(integrated_quality['improvement_areas'], list)
    
    @pytest.mark.asyncio
    async def test_integration_data_generation(self, scorer, sample_accuracy_score):
        """Test integration data generation for quality systems."""
        
        test_query = "What are glucose levels in diabetes?"
        test_response = "Glucose levels were 150 mg/dL in diabetic patients."
        
        integration_data = await scorer._generate_integration_data(
            sample_accuracy_score, test_query, test_response
        )
        
        required_fields = [
            'factual_accuracy_score', 'accuracy_grade', 'reliability_indicator',
            'dimension_scores', 'integration_weights', 'quality_boost_eligible',
            'performance_indicators'
        ]
        
        for field in required_fields:
            assert field in integration_data
        
        # Check relevance scorer compatibility if enabled
        if scorer.config['integration_settings']['enable_relevance_integration']:
            assert 'relevance_scorer_compatibility' in integration_data
            compatibility = integration_data['relevance_scorer_compatibility']
            assert 'dimension_scores' in compatibility
            assert 'overall_adjustment_factor' in compatibility
            assert 'integration_weight' in compatibility
    
    @pytest.mark.asyncio
    async def test_combined_quality_score_calculation(self, scorer, sample_accuracy_score, mock_relevance_scorer):
        """Test combined quality score calculation."""
        
        # Mock relevance score
        relevance_score = Mock()
        relevance_score.overall_score = 76.5
        relevance_score.confidence_score = 78.0
        
        combined_score = scorer._calculate_combined_quality_score(
            sample_accuracy_score, relevance_score
        )
        
        assert 0 <= combined_score <= 100
        assert isinstance(combined_score, float)
        
        # Test quality boost scenario
        high_accuracy_score = Mock()
        high_accuracy_score.overall_score = 92.0
        high_accuracy_score.confidence_score = 90.0
        
        high_relevance_score = Mock()
        high_relevance_score.overall_score = 88.0
        high_relevance_score.confidence_score = 87.0
        
        boosted_score = scorer._calculate_combined_quality_score(
            high_accuracy_score, high_relevance_score
        )
        
        # Should receive quality boost
        non_boosted_score = scorer._calculate_combined_quality_score(
            sample_accuracy_score, relevance_score
        )
        assert boosted_score >= non_boosted_score
    
    # Performance Tests
    
    @pytest.mark.asyncio
    async def test_scoring_performance(self, scorer, performance_test_data, performance_monitor):
        """Test scoring performance with large datasets."""
        
        many_results = []
        for claim in performance_test_data['many_claims']:
            result = Mock(spec=VerificationResult)
            result.claim_id = claim.claim_id
            result.verification_status = VerificationStatus.SUPPORTED
            result.verification_confidence = 85.0
            result.evidence_strength = 80.0
            result.context_match = 82.0
            result.supporting_evidence = []
            result.contradicting_evidence = []
            result.neutral_evidence = []
            result.total_evidence_count = 2
            result.processing_time_ms = 75.0
            result.verification_strategy = claim.claim_type
            result.metadata = {'claim_type': claim.claim_type}
            many_results.append(result)
        
        performance_monitor.start_measurement("batch_scoring")
        
        accuracy_score = await scorer.score_accuracy(many_results)
        
        processing_time = performance_monitor.end_measurement("batch_scoring")
        
        assert accuracy_score is not None
        assert accuracy_score.total_claims_assessed == len(many_results)
        assert processing_time < performance_test_data['performance_thresholds']['batch_verification_ms']
    
    @pytest.mark.asyncio
    async def test_report_generation_performance(self, scorer, sample_verification_results, performance_monitor):
        """Test report generation performance."""
        
        performance_monitor.start_measurement("report_generation")
        
        report = await scorer.generate_comprehensive_report(sample_verification_results)
        
        processing_time = performance_monitor.end_measurement("report_generation")
        
        assert report is not None
        assert isinstance(report, AccuracyReport)
        assert processing_time < 2000  # Should complete within 2 seconds
    
    # Error Handling Tests
    
    @pytest.mark.asyncio
    async def test_scoring_error_handling(self, scorer, error_test_scenarios):
        """Test error handling in scoring process."""
        
        # Test with malformed verification results
        malformed_results = []
        for i in range(3):
            result = Mock()
            result.claim_id = f"malformed_{i}"
            result.verification_status = None  # Invalid status
            result.verification_confidence = -50  # Invalid confidence
            result.evidence_strength = 150  # Invalid strength (>100)
            result.supporting_evidence = None  # Should be list
            malformed_results.append(result)
        
        # Should handle errors gracefully
        try:
            accuracy_score = await scorer.score_accuracy(malformed_results)
            # If it doesn't raise an exception, check for default values
            assert accuracy_score.overall_score >= 0
        except AccuracyScoringError:
            # Expected error handling
            pass
    
    @pytest.mark.asyncio
    async def test_report_generation_error_handling(self, scorer):
        """Test error handling in report generation."""
        
        # Test with None verification results
        with pytest.raises(ReportGenerationError):
            await scorer.generate_comprehensive_report(None)
    
    @pytest.mark.asyncio
    async def test_integration_error_handling(self, scorer, sample_accuracy_score):
        """Test error handling in integration processes."""
        
        # Test integration without relevance scorer
        scorer.config['integration_settings']['enable_relevance_integration'] = False
        
        with pytest.raises(QualityIntegrationError):
            await scorer.integrate_with_relevance_scorer(
                sample_accuracy_score, "test query", "test response"
            )
    
    # Configuration and Customization Tests
    
    @pytest.mark.asyncio
    async def test_custom_scoring_weights(self, scorer_config):
        """Test custom scoring weight configurations."""
        
        # Test extreme weight configurations
        extreme_config = scorer_config.copy()
        extreme_config['scoring_weights'] = {
            'claim_verification': 0.9,
            'evidence_quality': 0.05,
            'coverage_assessment': 0.03,
            'consistency_analysis': 0.01,
            'confidence_factor': 0.01
        }
        
        scorer = FactualAccuracyScorer(config=extreme_config)
        
        assert scorer.scoring_weights == extreme_config['scoring_weights']
        assert sum(scorer.scoring_weights.values()) == pytest.approx(1.0)
    
    @pytest.mark.asyncio
    async def test_custom_claim_type_weights(self, scorer_config, sample_verification_results):
        """Test custom claim type weight configurations."""
        
        custom_config = scorer_config.copy()
        custom_config['claim_type_weights'] = {
            'numeric': 2.0,  # Very high weight
            'qualitative': 0.5,  # Very low weight
            'methodological': 1.5
        }
        
        scorer = FactualAccuracyScorer(config=custom_config)
        
        accuracy_score = await scorer.score_accuracy(sample_verification_results)
        
        # Numeric claims should be heavily weighted
        if 'numeric' in accuracy_score.claim_type_scores:
            numeric_score = accuracy_score.claim_type_scores['numeric']
            assert numeric_score > 0  # Should be boosted by high weight
    
    @pytest.mark.asyncio
    async def test_threshold_configurations(self, scorer_config):
        """Test custom threshold configurations."""
        
        custom_config = scorer_config.copy()
        custom_config['evidence_quality_thresholds'] = {
            'high_quality': 95.0,  # Very high threshold
            'medium_quality': 75.0,
            'low_quality': 50.0
        }
        
        scorer = FactualAccuracyScorer(config=custom_config)
        
        assert scorer.config['evidence_quality_thresholds']['high_quality'] == 95.0
    
    # Statistics and Monitoring Tests
    
    @pytest.mark.asyncio
    async def test_scoring_statistics_tracking(self, scorer, sample_verification_results):
        """Test scoring statistics tracking."""
        
        initial_stats = scorer.get_scoring_statistics()
        assert initial_stats['total_scorings'] == 0
        assert initial_stats['total_claims_scored'] == 0
        
        # Perform scoring operations
        for _ in range(3):
            await scorer.score_accuracy(sample_verification_results)
        
        final_stats = scorer.get_scoring_statistics()
        assert final_stats['total_scorings'] == 3
        assert final_stats['total_claims_scored'] == 3 * len(sample_verification_results)
        assert final_stats['average_claims_per_scoring'] > 0
        assert 'processing_times' in final_stats
    
    @pytest.mark.asyncio
    async def test_processing_time_tracking(self, scorer, sample_verification_results):
        """Test processing time tracking accuracy."""
        
        start_time = time.time()
        accuracy_score = await scorer.score_accuracy(sample_verification_results)
        end_time = time.time()
        
        actual_processing_time = (end_time - start_time) * 1000
        reported_processing_time = accuracy_score.processing_time_ms
        
        # Should be within reasonable margin
        assert abs(actual_processing_time - reported_processing_time) < 100  # 100ms margin
        assert reported_processing_time > 0
    
    # Utility Method Tests
    
    def test_accuracy_score_properties(self, sample_accuracy_score):
        """Test AccuracyScore properties and methods."""
        
        assert sample_accuracy_score.accuracy_percentage == "84.7%"
        assert sample_accuracy_score.is_reliable is True  # Score > 70%
        
        # Test dictionary conversion
        score_dict = sample_accuracy_score.to_dict()
        assert isinstance(score_dict, dict)
        assert 'overall_score' in score_dict
        assert score_dict['grade'] == sample_accuracy_score.grade.value
    
    def test_accuracy_report_properties(self):
        """Test AccuracyReport properties and methods."""
        
        report = AccuracyReport(
            report_id="test_report",
            accuracy_score=AccuracyScore(overall_score=85.0, grade=AccuracyGrade.GOOD)
        )
        
        assert "test_report" in report.report_summary
        assert "85.0%" in report.report_summary
        assert "Good" in report.report_summary
        
        # Test dictionary conversion
        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert 'report_id' in report_dict
        assert 'created_timestamp' in report_dict


# Convenience Function Tests
@pytest.mark.accuracy_scorer
class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    @pytest.mark.asyncio
    async def test_score_verification_results_function(self, sample_verification_results, sample_extracted_claims):
        """Test score_verification_results convenience function."""
        
        result = await score_verification_results(
            sample_verification_results, 
            claims=sample_extracted_claims
        )
        
        assert isinstance(result, AccuracyScore)
        assert result.total_claims_assessed == len(sample_verification_results)
    
    @pytest.mark.asyncio
    async def test_generate_accuracy_report_function(self, sample_verification_results, sample_extracted_claims):
        """Test generate_accuracy_report convenience function."""
        
        report = await generate_accuracy_report(
            sample_verification_results,
            claims=sample_extracted_claims,
            query="test query",
            response="test response"
        )
        
        assert isinstance(report, AccuracyReport)
        assert report.accuracy_score.total_claims_assessed == len(sample_verification_results)
    
    @pytest.mark.asyncio
    async def test_integrate_quality_assessment_function(self, sample_verification_results, sample_extracted_claims):
        """Test integrate_quality_assessment convenience function."""
        
        with patch('lightrag_integration.accuracy_scorer.ClinicalMetabolomicsRelevanceScorer') as mock_relevance_class:
            mock_relevance_scorer = Mock()
            mock_relevance_score = Mock()
            mock_relevance_score.overall_score = 78.0
            mock_relevance_score.confidence_score = 75.0
            mock_relevance_scorer.calculate_relevance_score = AsyncMock(return_value=mock_relevance_score)
            mock_relevance_class.return_value = mock_relevance_scorer
            
            result = await integrate_quality_assessment(
                sample_verification_results,
                "test query",
                "test response",
                claims=sample_extracted_claims
            )
            
            assert isinstance(result, dict)
            assert 'factual_accuracy' in result
            assert 'integrated_quality' in result


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short", "-m", "accuracy_scorer"])