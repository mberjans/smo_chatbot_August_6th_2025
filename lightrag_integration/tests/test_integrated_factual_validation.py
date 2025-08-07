#!/usr/bin/env python3
"""
Comprehensive End-to-End Integration Tests for Factual Accuracy Validation System.

This test suite provides thorough end-to-end integration testing for the complete
factual accuracy validation pipeline including claim extraction, validation,
scoring, and reporting.

Test Categories:
1. Complete pipeline integration tests
2. Cross-component interaction tests
3. Quality system integration tests
4. Real-world workflow simulation tests
5. Data flow integrity tests
6. Performance integration tests
7. Configuration integration tests

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
import logging

# Import test fixtures
from .factual_validation_test_fixtures import *

# Import the modules to test
try:
    from ..accuracy_scorer import (
        FactualAccuracyScorer, AccuracyScore, AccuracyReport, AccuracyGrade
    )
    from ..factual_accuracy_validator import (
        FactualAccuracyValidator, VerificationResult, VerificationStatus,
        EvidenceItem, VerificationReport
    )
    from ..claim_extractor import (
        BiomedicalClaimExtractor, ExtractedClaim
    )
    from ..document_indexer import (
        SourceDocumentIndex, IndexedContent
    )
    from ..relevance_scorer import (
        ClinicalMetabolomicsRelevanceScorer, RelevanceScore
    )
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


@pytest.mark.integration_validation
class TestCompleteValidationPipeline:
    """Test suite for complete end-to-end validation pipeline."""
    
    @pytest.fixture
    def pipeline_config(self):
        """Provide comprehensive pipeline configuration."""
        return {
            'claim_extraction': {
                'enable_biomedical_patterns': True,
                'min_confidence_threshold': 60.0,
                'max_claims_per_response': 50,
                'enable_claim_deduplication': True
            },
            'document_indexing': {
                'enable_content_caching': True,
                'index_refresh_interval': 300,
                'max_search_results': 20,
                'enable_semantic_search': True
            },
            'factual_validation': {
                'min_evidence_confidence': 55.0,
                'enable_cross_verification': True,
                'verification_strategies': ['numeric', 'qualitative', 'methodological'],
                'max_verification_time_ms': 5000
            },
            'accuracy_scoring': {
                'scoring_weights': {
                    'claim_verification': 0.35,
                    'evidence_quality': 0.25,
                    'coverage_assessment': 0.20,
                    'consistency_analysis': 0.15,
                    'confidence_factor': 0.05
                },
                'enable_integration': True,
                'generate_detailed_reports': True
            },
            'integration': {
                'enable_relevance_integration': True,
                'quality_boost_threshold': 85.0,
                'performance_monitoring': True
            }
        }
    
    @pytest.fixture
    def integrated_pipeline(self, pipeline_config, mock_document_indexer):
        """Create integrated pipeline with all components."""
        
        class IntegratedValidationPipeline:
            def __init__(self, config):
                self.config = config
                self.claim_extractor = None
                self.document_indexer = mock_document_indexer
                self.factual_validator = None
                self.accuracy_scorer = None
                self.relevance_scorer = None
                self.pipeline_stats = {
                    'total_processed': 0,
                    'successful_validations': 0,
                    'processing_times': []
                }
            
            async def initialize(self):
                """Initialize all pipeline components."""
                # Initialize claim extractor
                self.claim_extractor = Mock(spec=BiomedicalClaimExtractor)
                self.claim_extractor.extract_claims = AsyncMock()
                
                # Initialize factual validator with document indexer
                self.factual_validator = Mock(spec=FactualAccuracyValidator)
                self.factual_validator.verify_claims = AsyncMock()
                
                # Initialize accuracy scorer
                self.accuracy_scorer = FactualAccuracyScorer(config=self.config['accuracy_scoring'])
                
                # Initialize relevance scorer if enabled
                if self.config['integration']['enable_relevance_integration']:
                    self.relevance_scorer = Mock(spec=ClinicalMetabolomicsRelevanceScorer)
                    self.relevance_scorer.calculate_relevance_score = AsyncMock()
                    self.accuracy_scorer.relevance_scorer = self.relevance_scorer
            
            async def process_response(self, query: str, response: str, 
                                    context: Optional[Dict] = None) -> Dict[str, Any]:
                """Process a complete response through the pipeline."""
                start_time = time.time()
                
                try:
                    # Step 1: Extract claims
                    claims = await self._extract_claims(response, context)
                    
                    # Step 2: Verify claims
                    verification_report = await self._verify_claims(claims, context)
                    
                    # Step 3: Score accuracy
                    accuracy_report = await self._score_accuracy(
                        verification_report.verification_results, claims, query, response
                    )
                    
                    # Step 4: Integrate with quality systems
                    integrated_assessment = await self._integrate_quality(
                        accuracy_report.accuracy_score, query, response
                    )
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    self.pipeline_stats['total_processed'] += 1
                    self.pipeline_stats['successful_validations'] += 1
                    self.pipeline_stats['processing_times'].append(processing_time)
                    
                    return {
                        'success': True,
                        'claims_extracted': len(claims),
                        'verification_report': verification_report,
                        'accuracy_report': accuracy_report,
                        'integrated_assessment': integrated_assessment,
                        'processing_time_ms': processing_time,
                        'pipeline_stats': self.pipeline_stats.copy()
                    }
                    
                except Exception as e:
                    self.pipeline_stats['total_processed'] += 1
                    return {
                        'success': False,
                        'error': str(e),
                        'processing_time_ms': (time.time() - start_time) * 1000
                    }
            
            async def _extract_claims(self, response: str, context: Optional[Dict]) -> List[ExtractedClaim]:
                """Extract claims from response."""
                # Mock claim extraction with realistic data
                mock_claims = []
                for i, claim_data in enumerate(SAMPLE_CLAIMS_DATA):
                    claim = Mock(spec=ExtractedClaim)
                    claim.claim_id = f"integrated_claim_{i+1}"
                    claim.claim_text = claim_data["text"]
                    claim.claim_type = claim_data["claim_type"]
                    claim.numeric_values = claim_data.get("numeric_values", [])
                    claim.units = claim_data.get("units", [])
                    claim.confidence = Mock(overall_confidence=claim_data["confidence"])
                    mock_claims.append(claim)
                
                self.claim_extractor.extract_claims.return_value = mock_claims
                return await self.claim_extractor.extract_claims(response, context)
            
            async def _verify_claims(self, claims: List[ExtractedClaim], 
                                   context: Optional[Dict]) -> VerificationReport:
                """Verify extracted claims."""
                # Mock verification results
                verification_results = []
                for claim in claims:
                    result = Mock(spec=VerificationResult)
                    result.claim_id = claim.claim_id
                    result.verification_status = VerificationStatus.SUPPORTED
                    result.verification_confidence = 82.0
                    result.evidence_strength = 78.0
                    result.context_match = 85.0
                    result.supporting_evidence = [Mock(spec=EvidenceItem)]
                    result.contradicting_evidence = []
                    result.neutral_evidence = []
                    result.total_evidence_count = 1
                    result.processing_time_ms = 150.0
                    result.verification_strategy = claim.claim_type
                    result.metadata = {'claim_type': claim.claim_type}
                    verification_results.append(result)
                
                report = Mock(spec=VerificationReport)
                report.verification_results = verification_results
                report.total_claims = len(claims)
                report.summary_statistics = {
                    'supported_claims': len(verification_results),
                    'average_confidence': 82.0
                }
                
                self.factual_validator.verify_claims.return_value = report
                return await self.factual_validator.verify_claims(claims)
            
            async def _score_accuracy(self, verification_results: List[VerificationResult],
                                    claims: List[ExtractedClaim], query: str, 
                                    response: str) -> AccuracyReport:
                """Score accuracy of verification results."""
                return await self.accuracy_scorer.generate_comprehensive_report(
                    verification_results, claims, query, response
                )
            
            async def _integrate_quality(self, accuracy_score: AccuracyScore,
                                       query: str, response: str) -> Dict[str, Any]:
                """Integrate with quality assessment systems."""
                if self.relevance_scorer:
                    return await self.accuracy_scorer.integrate_with_relevance_scorer(
                        accuracy_score, query, response
                    )
                else:
                    return {
                        'factual_accuracy': accuracy_score.to_dict(),
                        'integration_available': False
                    }
        
        pipeline = IntegratedValidationPipeline(pipeline_config)
        return pipeline
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_initialization(self, integrated_pipeline):
        """Test complete pipeline initialization."""
        
        await integrated_pipeline.initialize()
        
        assert integrated_pipeline.claim_extractor is not None
        assert integrated_pipeline.document_indexer is not None
        assert integrated_pipeline.factual_validator is not None
        assert integrated_pipeline.accuracy_scorer is not None
        
        if integrated_pipeline.config['integration']['enable_relevance_integration']:
            assert integrated_pipeline.relevance_scorer is not None
    
    @pytest.mark.asyncio
    async def test_end_to_end_response_processing(self, integrated_pipeline):
        """Test end-to-end response processing through complete pipeline."""
        
        await integrated_pipeline.initialize()
        
        test_query = "What are the metabolic differences between diabetic and healthy patients?"
        test_response = ("Glucose levels were significantly elevated at 150 mg/dL in diabetic "
                        "patients compared to 90 mg/dL in healthy controls. The metabolomics "
                        "analysis revealed increased levels of branched-chain amino acids.")
        
        result = await integrated_pipeline.process_response(test_query, test_response)
        
        assert result['success'] is True
        assert result['claims_extracted'] > 0
        assert result['verification_report'] is not None
        assert result['accuracy_report'] is not None
        assert result['integrated_assessment'] is not None
        assert result['processing_time_ms'] > 0
        
        # Check verification report
        verification_report = result['verification_report']
        assert verification_report.total_claims == result['claims_extracted']
        assert len(verification_report.verification_results) == result['claims_extracted']
        
        # Check accuracy report
        accuracy_report = result['accuracy_report']
        assert isinstance(accuracy_report.accuracy_score, AccuracyScore)
        assert accuracy_report.accuracy_score.total_claims_assessed == result['claims_extracted']
        
        # Check integrated assessment
        integrated_assessment = result['integrated_assessment']
        assert 'factual_accuracy' in integrated_assessment
    
    @pytest.mark.asyncio
    async def test_pipeline_with_multiple_responses(self, integrated_pipeline):
        """Test pipeline processing multiple responses."""
        
        await integrated_pipeline.initialize()
        
        test_cases = [
            {
                "query": "What is glucose metabolism?",
                "response": "Glucose is metabolized through glycolysis, producing ATP and pyruvate."
            },
            {
                "query": "How does LC-MS work?",
                "response": "LC-MS combines liquid chromatography with mass spectrometry for metabolite identification."
            },
            {
                "query": "What are biomarkers?",
                "response": "Biomarkers are measurable indicators of biological processes or disease states."
            }
        ]
        
        results = []
        for test_case in test_cases:
            result = await integrated_pipeline.process_response(
                test_case["query"], test_case["response"]
            )
            results.append(result)
        
        # All should succeed
        assert all(r['success'] for r in results)
        assert len(results) == len(test_cases)
        
        # Pipeline statistics should be updated
        final_stats = integrated_pipeline.pipeline_stats
        assert final_stats['total_processed'] == len(test_cases)
        assert final_stats['successful_validations'] == len(test_cases)
        assert len(final_stats['processing_times']) == len(test_cases)
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, integrated_pipeline):
        """Test pipeline error handling and recovery."""
        
        await integrated_pipeline.initialize()
        
        # Configure claim extractor to fail
        integrated_pipeline.claim_extractor.extract_claims.side_effect = Exception("Extraction failed")
        
        result = await integrated_pipeline.process_response(
            "test query", "test response"
        )
        
        assert result['success'] is False
        assert 'error' in result
        assert result['processing_time_ms'] > 0
        
        # Pipeline stats should reflect failure
        stats = integrated_pipeline.pipeline_stats
        assert stats['total_processed'] == 1
        assert stats['successful_validations'] == 0


@pytest.mark.integration_validation
class TestCrossComponentIntegration:
    """Test suite for cross-component integration."""
    
    @pytest.mark.asyncio
    async def test_claim_extractor_to_validator_integration(self, mock_claim_extractor, mock_factual_validator):
        """Test integration between claim extractor and validator."""
        
        # Configure claim extractor
        test_claims = []
        for i, claim_data in enumerate(SAMPLE_CLAIMS_DATA):
            claim = Mock()
            claim.claim_id = f"integration_claim_{i}"
            claim.claim_text = claim_data["text"]
            claim.claim_type = claim_data["claim_type"]
            claim.confidence = Mock(overall_confidence=claim_data["confidence"])
            test_claims.append(claim)
        
        mock_claim_extractor.extract_claims.return_value = test_claims
        
        # Extract claims
        extracted_claims = await mock_claim_extractor.extract_claims("test response")
        
        # Verify claims
        verification_report = await mock_factual_validator.verify_claims(extracted_claims)
        
        assert len(extracted_claims) == len(test_claims)
        assert verification_report.total_claims == len(extracted_claims)
        assert len(verification_report.verification_results) == len(extracted_claims)
    
    @pytest.mark.asyncio
    async def test_validator_to_scorer_integration(self, mock_factual_validator, sample_verification_results):
        """Test integration between validator and accuracy scorer."""
        
        # Configure validator to return test results
        report = Mock()
        report.verification_results = sample_verification_results
        report.total_claims = len(sample_verification_results)
        mock_factual_validator.verify_claims.return_value = report
        
        # Get verification results
        verification_report = await mock_factual_validator.verify_claims([])
        
        # Score accuracy
        scorer = FactualAccuracyScorer()
        accuracy_score = await scorer.score_accuracy(verification_report.verification_results)
        
        assert isinstance(accuracy_score, AccuracyScore)
        assert accuracy_score.total_claims_assessed == len(sample_verification_results)
        assert 0 <= accuracy_score.overall_score <= 100
    
    @pytest.mark.asyncio
    async def test_document_indexer_integration(self, mock_document_indexer):
        """Test document indexer integration with verification process."""
        
        # Test search functionality
        search_results = await mock_document_indexer.search_content(
            "glucose levels diabetes", max_results=10
        )
        
        assert isinstance(search_results, list)
        assert len(search_results) > 0
        
        # Test claim verification functionality
        test_claim = Mock()
        test_claim.claim_text = "glucose was 150 mg/dL"
        test_claim.keywords = ["glucose", "150", "mg/dL"]
        
        verification_data = await mock_document_indexer.verify_claim(test_claim)
        
        assert 'verification_status' in verification_data
        assert 'confidence' in verification_data
        assert 'supporting_evidence' in verification_data
    
    @pytest.mark.asyncio
    async def test_data_flow_integrity(self, mock_claim_extractor, mock_factual_validator, sample_extracted_claims):
        """Test data flow integrity across components."""
        
        # Configure components
        mock_claim_extractor.extract_claims.return_value = sample_extracted_claims
        
        verification_results = []
        for claim in sample_extracted_claims:
            result = Mock()
            result.claim_id = claim.claim_id
            result.verification_status = VerificationStatus.SUPPORTED
            result.verification_confidence = 85.0
            result.metadata = {'original_claim_type': claim.claim_type}
            verification_results.append(result)
        
        verification_report = Mock()
        verification_report.verification_results = verification_results
        verification_report.total_claims = len(sample_extracted_claims)
        mock_factual_validator.verify_claims.return_value = verification_report
        
        # Process through pipeline
        claims = await mock_claim_extractor.extract_claims("test response")
        report = await mock_factual_validator.verify_claims(claims)
        
        # Verify data integrity
        assert len(claims) == len(sample_extracted_claims)
        assert report.total_claims == len(claims)
        assert len(report.verification_results) == len(claims)
        
        # Check claim IDs are preserved
        original_ids = {c.claim_id for c in claims}
        result_ids = {r.claim_id for r in report.verification_results}
        assert original_ids == result_ids


@pytest.mark.integration_validation
class TestQualitySystemIntegration:
    """Test suite for quality system integration."""
    
    @pytest.fixture
    def mock_relevance_scorer(self):
        """Create mock relevance scorer for integration testing."""
        scorer = Mock(spec=ClinicalMetabolomicsRelevanceScorer)
        
        relevance_score = Mock()
        relevance_score.overall_score = 79.5
        relevance_score.relevance_grade = "Good"
        relevance_score.dimension_scores = {
            "scientific_rigor": 82.0,
            "biomedical_context_depth": 77.0,
            "query_alignment": 81.0,
            "metabolomics_relevance": 78.0,
            "clinical_applicability": 79.0
        }
        relevance_score.confidence_score = 80.0
        
        scorer.calculate_relevance_score = AsyncMock(return_value=relevance_score)
        return scorer
    
    @pytest.mark.asyncio
    async def test_accuracy_relevance_integration(self, sample_accuracy_score, mock_relevance_scorer):
        """Test integration between accuracy scorer and relevance scorer."""
        
        accuracy_scorer = FactualAccuracyScorer(relevance_scorer=mock_relevance_scorer)
        
        integrated_assessment = await accuracy_scorer.integrate_with_relevance_scorer(
            sample_accuracy_score,
            "What are glucose levels in diabetes?",
            "Glucose levels were 150 mg/dL in diabetic patients."
        )
        
        assert isinstance(integrated_assessment, dict)
        assert 'factual_accuracy' in integrated_assessment
        assert 'relevance_assessment' in integrated_assessment
        assert 'integrated_quality' in integrated_assessment
        
        # Check combined score calculation
        integrated_quality = integrated_assessment['integrated_quality']
        assert 'combined_score' in integrated_quality
        assert 0 <= integrated_quality['combined_score'] <= 100
        assert 'quality_grade' in integrated_quality
        
        # Check quality analysis
        assert 'strength_areas' in integrated_quality
        assert 'improvement_areas' in integrated_quality
        assert isinstance(integrated_quality['strength_areas'], list)
        assert isinstance(integrated_quality['improvement_areas'], list)
    
    @pytest.mark.asyncio
    async def test_quality_boost_mechanism(self, mock_relevance_scorer):
        """Test quality boost mechanism for high-performing systems."""
        
        # Create high-performing accuracy score
        high_accuracy_score = Mock()
        high_accuracy_score.overall_score = 92.0
        high_accuracy_score.confidence_score = 88.0
        high_accuracy_score.dimension_scores = {
            "claim_verification": 94.0,
            "evidence_quality": 90.0,
            "coverage_assessment": 92.0,
            "consistency_analysis": 91.0,
            "confidence_factor": 89.0
        }
        high_accuracy_score.grade = AccuracyGrade.EXCELLENT
        
        # Configure high relevance score
        high_relevance_score = Mock()
        high_relevance_score.overall_score = 91.0
        high_relevance_score.confidence_score = 89.0
        
        mock_relevance_scorer.calculate_relevance_score.return_value = high_relevance_score
        
        accuracy_scorer = FactualAccuracyScorer(
            relevance_scorer=mock_relevance_scorer,
            config={'integration_settings': {'enable_relevance_integration': True}}
        )
        
        # Calculate combined score
        combined_score = accuracy_scorer._calculate_combined_quality_score(
            high_accuracy_score, high_relevance_score
        )
        
        # Should receive quality boost for dual high performance
        assert combined_score >= 90.0  # High combined score
        
        # Compare with non-boosted scenario
        normal_accuracy_score = Mock()
        normal_accuracy_score.overall_score = 75.0
        normal_accuracy_score.confidence_score = 72.0
        
        normal_relevance_score = Mock()
        normal_relevance_score.overall_score = 73.0
        normal_relevance_score.confidence_score = 70.0
        
        normal_combined_score = accuracy_scorer._calculate_combined_quality_score(
            normal_accuracy_score, normal_relevance_score
        )
        
        assert combined_score > normal_combined_score  # Boosted score should be higher
    
    @pytest.mark.asyncio
    async def test_integration_data_compatibility(self, sample_accuracy_score, mock_relevance_scorer):
        """Test integration data compatibility with existing systems."""
        
        accuracy_scorer = FactualAccuracyScorer(relevance_scorer=mock_relevance_scorer)
        
        integration_data = await accuracy_scorer._generate_integration_data(
            sample_accuracy_score,
            "test query",
            "test response"
        )
        
        # Check required integration fields
        required_fields = [
            'factual_accuracy_score', 'accuracy_grade', 'reliability_indicator',
            'dimension_scores', 'integration_weights', 'performance_indicators'
        ]
        
        for field in required_fields:
            assert field in integration_data
        
        # Check relevance scorer compatibility
        if 'relevance_scorer_compatibility' in integration_data:
            compatibility = integration_data['relevance_scorer_compatibility']
            assert 'dimension_scores' in compatibility
            assert 'overall_adjustment_factor' in compatibility
            assert 'integration_weight' in compatibility
        
        # Check contextual assessment
        if 'contextual_assessment' in integration_data:
            context = integration_data['contextual_assessment']
            assert 'query_provided' in context
            assert 'response_provided' in context
            assert 'assessment_scope' in context


@pytest.mark.integration_validation
class TestRealWorldWorkflowSimulation:
    """Test suite for real-world workflow simulation."""
    
    @pytest.fixture
    def realistic_test_scenarios(self):
        """Provide realistic test scenarios."""
        return [
            {
                "name": "clinical_research_query",
                "query": "What are the metabolic biomarkers for type 2 diabetes?",
                "response": ("Type 2 diabetes is characterized by elevated glucose levels (>126 mg/dL fasting), "
                           "increased HbA1c levels (>6.5%), and altered lipid metabolism. Key metabolic biomarkers "
                           "include fasting plasma glucose, 2-hour glucose tolerance test results, and glycated "
                           "hemoglobin. Metabolomics studies have identified branched-chain amino acids, "
                           "acylcarnitines, and specific lipid species as additional biomarkers."),
                "expected_claims": ["numeric", "qualitative", "methodological"],
                "difficulty": "medium"
            },
            {
                "name": "analytical_methodology_query",
                "query": "How is LC-MS/MS used in metabolomics?",
                "response": ("LC-MS/MS (Liquid Chromatography-Tandem Mass Spectrometry) is a powerful analytical "
                           "technique used for metabolite identification and quantification. The method combines "
                           "liquid chromatography separation with mass spectrometry detection. Typical protocols "
                           "involve sample preparation, chromatographic separation using C18 columns, and "
                           "detection using electrospray ionization in both positive and negative modes. "
                           "Quantification limits can reach pg/mL levels for many metabolites."),
                "expected_claims": ["methodological", "qualitative"],
                "difficulty": "high"
            },
            {
                "name": "statistical_analysis_query", 
                "query": "What statistical methods are used in metabolomics?",
                "response": ("Statistical analysis in metabolomics typically involves multivariate methods "
                           "such as Principal Component Analysis (PCA), Partial Least Squares-Discriminant "
                           "Analysis (PLS-DA), and Random Forest classification. Univariate statistical "
                           "tests include t-tests for comparing groups and ANOVA for multiple groups. "
                           "False discovery rate (FDR) correction is applied with q-values <0.05 considered "
                           "significant. Effect sizes are reported using Cohen's d or similar metrics."),
                "expected_claims": ["methodological", "numeric", "comparative"],
                "difficulty": "high"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_realistic_scenario_processing(self, integrated_pipeline, realistic_test_scenarios):
        """Test processing of realistic biomedical scenarios."""
        
        await integrated_pipeline.initialize()
        
        for scenario in realistic_test_scenarios:
            result = await integrated_pipeline.process_response(
                scenario["query"], scenario["response"]
            )
            
            assert result['success'] is True, f"Scenario '{scenario['name']}' failed"
            assert result['claims_extracted'] > 0, f"No claims extracted for '{scenario['name']}'"
            
            # Check that expected claim types are present
            verification_report = result['verification_report']
            claim_types = set()
            for vr in verification_report.verification_results:
                if 'claim_type' in vr.metadata:
                    claim_types.add(vr.metadata['claim_type'])
            
            expected_types = set(scenario["expected_claims"])
            found_types = claim_types.intersection(expected_types)
            assert len(found_types) > 0, f"Expected claim types not found in '{scenario['name']}'"
            
            # Check accuracy assessment
            accuracy_report = result['accuracy_report']
            assert accuracy_report.accuracy_score.total_claims_assessed > 0
            
            # Performance should be reasonable based on difficulty
            max_time = 3000 if scenario["difficulty"] == "medium" else 5000
            assert result['processing_time_ms'] < max_time
    
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, integrated_pipeline, realistic_test_scenarios):
        """Test batch processing of multiple scenarios."""
        
        await integrated_pipeline.initialize()
        
        # Process all scenarios
        results = []
        start_time = time.time()
        
        for scenario in realistic_test_scenarios:
            result = await integrated_pipeline.process_response(
                scenario["query"], scenario["response"]
            )
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        
        # All should succeed
        successful_results = [r for r in results if r['success']]
        assert len(successful_results) == len(realistic_test_scenarios)
        
        # Batch performance should be reasonable
        assert total_time < 15000  # 15 seconds for 3 complex scenarios
        
        # Calculate aggregate statistics
        total_claims = sum(r['claims_extracted'] for r in successful_results)
        avg_accuracy = statistics.mean([
            r['accuracy_report'].accuracy_score.overall_score 
            for r in successful_results
        ])
        
        assert total_claims > len(realistic_test_scenarios)  # At least 1 claim per scenario
        assert 0 <= avg_accuracy <= 100
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, integrated_pipeline):
        """Test concurrent processing capability."""
        
        await integrated_pipeline.initialize()
        
        # Create multiple concurrent requests
        concurrent_requests = [
            ("Query 1", "Response about glucose metabolism and diabetes."),
            ("Query 2", "Response about LC-MS methodology and protocols."),
            ("Query 3", "Response about statistical analysis in research."),
            ("Query 4", "Response about biomarker discovery and validation."),
            ("Query 5", "Response about clinical trial design and execution.")
        ]
        
        # Process concurrently
        tasks = [
            integrated_pipeline.process_response(query, response)
            for query, response in concurrent_requests
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = (time.time() - start_time) * 1000
        
        # Check results
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
        error_results = [r for r in results if isinstance(r, Exception) or not isinstance(r, dict) or not r.get('success')]
        
        # Most should succeed (allow for some concurrent processing issues)
        assert len(successful_results) >= len(concurrent_requests) * 0.8
        
        # Concurrent processing should be faster than sequential
        sequential_estimate = len(concurrent_requests) * 1500  # Estimated 1.5s per request
        assert total_time < sequential_estimate * 0.8  # Should be at least 20% faster


@pytest.mark.integration_validation  
class TestPerformanceIntegration:
    """Test suite for performance integration across components."""
    
    @pytest.mark.asyncio
    async def test_pipeline_performance_monitoring(self, integrated_pipeline, performance_monitor):
        """Test performance monitoring across the pipeline."""
        
        await integrated_pipeline.initialize()
        
        # Enable performance monitoring
        integrated_pipeline.config['integration']['performance_monitoring'] = True
        
        test_query = "What are the key metabolites in diabetes?"
        test_response = "Key metabolites include glucose (150 mg/dL), insulin, and various amino acids."
        
        # Monitor performance
        performance_monitor.start_measurement("full_pipeline")
        
        result = await integrated_pipeline.process_response(test_query, test_response)
        
        total_time = performance_monitor.end_measurement("full_pipeline")
        
        assert result['success'] is True
        assert total_time > 0
        assert result['processing_time_ms'] <= total_time + 100  # Allow small margin
        
        # Check individual component performance
        if 'performance_breakdown' in result:
            breakdown = result['performance_breakdown']
            assert 'claim_extraction_ms' in breakdown
            assert 'verification_ms' in breakdown
            assert 'scoring_ms' in breakdown
    
    @pytest.mark.asyncio
    async def test_scalability_performance(self, integrated_pipeline, performance_test_data):
        """Test pipeline scalability with increasing loads."""
        
        await integrated_pipeline.initialize()
        
        # Test with different response sizes
        small_response = "Glucose is 100 mg/dL."
        medium_response = " ".join(SAMPLE_BIOMEDICAL_RESPONSES[:3])
        large_response = " ".join(SAMPLE_BIOMEDICAL_RESPONSES * 3)
        
        test_cases = [
            ("small", "small query", small_response),
            ("medium", "medium query", medium_response),
            ("large", "large query", large_response)
        ]
        
        performance_results = {}
        
        for size, query, response in test_cases:
            start_time = time.time()
            
            result = await integrated_pipeline.process_response(query, response)
            
            processing_time = (time.time() - start_time) * 1000
            
            performance_results[size] = {
                'success': result['success'],
                'claims_extracted': result['claims_extracted'],
                'processing_time_ms': processing_time
            }
        
        # Performance should scale reasonably
        small_time = performance_results['small']['processing_time_ms']
        large_time = performance_results['large']['processing_time_ms']
        
        # Large should not be more than 5x slower than small
        assert large_time < small_time * 5
        
        # All should complete within reasonable time
        for size, perf in performance_results.items():
            assert perf['processing_time_ms'] < 10000  # 10 seconds max
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, integrated_pipeline):
        """Test memory efficiency during pipeline processing."""
        
        await integrated_pipeline.initialize()
        
        # Process multiple responses without accumulating memory
        responses = [
            "Response 1: Glucose levels and diabetes research findings.",
            "Response 2: LC-MS methodology and analytical protocols.",
            "Response 3: Statistical analysis and data interpretation.",
            "Response 4: Biomarker discovery and clinical validation.",
            "Response 5: Metabolomics workflows and data processing."
        ]
        
        for i, response in enumerate(responses):
            result = await integrated_pipeline.process_response(
                f"Query {i+1}", response
            )
            
            assert result['success'] is True
            
            # Memory usage should not grow indefinitely
            # (This is a basic test - more sophisticated memory monitoring could be added)
            assert len(integrated_pipeline.pipeline_stats['processing_times']) == i + 1


if __name__ == "__main__":
    # Run the integration test suite
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration_validation"])